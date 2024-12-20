from __future__ import division, print_function

import copy
import math
import os
import re
import sys
import warnings
from contextlib import contextmanager

import matplotlib.pyplot as plt
import numpy as np
import photutils
import sep
import skimage.measure
import skimage.transform
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.table import Column, Table, vstack
from astropy.units import Quantity
from astropy.utils.exceptions import AstropyUserWarning
from astropy.visualization import (AsymmetricPercentileInterval, HistEqStretch,
                                   LogStretch, ZScaleInterval)
from astropy.visualization.mpl_normalize import ImageNormalize
from matplotlib import cm, colors, rcParams
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Ellipse
from matplotlib.patches import Ellipse as mplEllip
from matplotlib_scalebar.scalebar import ANGLE, ScaleBar
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from palettable.colorbrewer.sequential import (Blues_9, Greys_9, OrRd_9,
                                               Purples_9, YlGn_9)
from scipy import ndimage
from scipy.interpolate import interp1d
from scipy.special import gamma, gammaincinv

system_use = sys.platform

Cmap = 'cividis'
Cmap_r = 'cividis_r'
Vmap = 'viridis'
Vmap_r = 'viridis_r'

red = '#FF4C00'
hedinghong = '#D24735'
green = '#21A675'
purple = '#8D4BBB'
blue = '#4B5CC4'
gold = '#F2BE45'
jianghuang = '#FFC773'
xiangye = '#ECD452'


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


def _set_matplotlib(style='default',
                    usetex=False,
                    fontsize=15,
                    figsize=(6, 5),
                    dpi=100):
    '''
    Default matplotlib settings, borrowed from Song Huang. I really like his plotting style.

    Parameters:
        style (str): options are "JL", "SM" (supermongo-like).
    '''
    # Use JL as a template
    # if style == 'default':
    #     plt.style.use('../mplstyle/default.mplstyle')
    # else:
    #     plt.style.use('../mplstyle/JL.mplstyle')
    import Taiji
    pkg_path = Taiji.__path__[0]
    if style == 'default':
        plt.style.use(os.path.join(pkg_path, 'mplstyle/default.mplstyle'))

    # Now it should be the _JL.mplstyle.
    elif style == 'JL':
        plt.style.use(os.path.join(pkg_path, 'mplstyle/JL.mplstyle'))

    elif style == 'DW':
        plt.style.use(['science', 'seaborn-colorblind'])

        plt.rcParams['figure.figsize'] = (10, 7)
        plt.rcParams['font.size'] = 25
        plt.rcParams['lines.linewidth'] = 3
        plt.rcParams['xtick.labelsize'] = 25
        plt.rcParams['ytick.labelsize'] = 25
        plt.rcParams['xtick.direction'] = 'in'
        plt.rcParams['ytick.direction'] = 'in'
        plt.rcParams['xtick.major.size'] = 8
        plt.rcParams['ytick.major.size'] = 8
        plt.rcParams['xtick.minor.size'] = 5
        plt.rcParams['ytick.minor.size'] = 5
        plt.rcParams['xtick.major.pad'] = 5
        plt.rcParams['xtick.minor.pad'] = 4.8
        plt.rcParams['ytick.major.pad'] = 5
        plt.rcParams['ytick.minor.pad'] = 4.8
        plt.rcParams['xtick.top'] = True
        plt.rcParams['ytick.right'] = True
        plt.rcParams['legend.frameon'] = True
        plt.rcParams['axes.labelpad'] = 8.0
        plt.rcParams['figure.constrained_layout.h_pad'] = 0
        plt.rcParams['text.usetex'] = True
        plt.rc('text', usetex=True)
        plt.rcParams['font.sans-serif'] = ['Times New Roman']
        plt.tick_params(axis='both', which='minor', labelsize=18)
        plt.rcParams['legend.edgecolor'] = 'black'
        # plt.rcParams['xtick.major.width'] = 3.8
        # plt.rcParams['xtick.minor.width'] = 3.2
        # plt.rcParams['ytick.major.width'] = 3.8
        # plt.rcParams['ytick.minor.width'] = 3.2
        # plt.rcParams['axes.linewidth'] = 5
        plt.close()

    elif style == 'nature':
        rcParams.update({
            "font.family": "sans-serif",
            # The default edge colors for scatter plots.
            "scatter.edgecolors": "black",
            "mathtext.fontset": "stixsans"
        })

    rcParams.update({
        'font.size': fontsize,
        'figure.figsize': "{0}, {1}".format(figsize[0], figsize[1]),
        'text.usetex': usetex,
        'figure.dpi': dpi,
        'legend.frameon': True,
        'figure.constrained_layout.h_pad': 0
    })


def set_matplotlib(style='default',
                   usetex=False,
                   fontsize=15,
                   figsize=(6, 5),
                   dpi=60):
    '''
    This function is from Kuaizi package developed by Jiaxuan Li.
    Default matplotlib settings, borrowed from Song Huang. I really like his plotting style.

    Parameters:
        style (str): options are "JL", "SM" (supermongo-like).
    '''

    import matplotlib.pyplot as plt
    from matplotlib import rcParams

    import Taiji

    # Use JL as a template
    pkg_path = Taiji.__path__[0]
    if style == 'default':
        plt.style.use(os.path.join(pkg_path, 'mplstyle/default.mplstyle'))
    elif style == 'JL':
        plt.style.use(os.path.join(pkg_path, 'mplstyle/JL.mplstyle'))

    rcParams.update({
        'font.size': fontsize,
        'figure.figsize': "{0}, {1}".format(figsize[0], figsize[1]),
        'text.usetex': usetex,
        'figure.dpi': dpi,
        'legend.frameon': True,
        'figure.constrained_layout.h_pad': 0
    })

    if style == 'SM':
        rcParams.update({
            "figure.figsize": "6, 6",
            "axes.linewidth": 0.6,
            "xtick.major.width": 0.5,
            "xtick.minor.width": 0.3,
            "ytick.major.width": 0.5,
            "ytick.minor.width": 0.3,
            "font.family": "monospace",
            "font.stretch": "semi-expanded",
            # The default edge colors for scatter plots.
            "scatter.edgecolors": "black",
            "mathtext.bf": "monospace:bold",
            "mathtext.cal": "monospace:bold",
            "mathtext.it": "monospace:italic",
            "mathtext.rm": "monospace",
            "mathtext.sf": "monospace",
            "mathtext.tt": "monospace",
            "mathtext.fallback": "cm",
            "mathtext.default": 'it'
        })

        if usetex is True:
            rcParams.update({
                "text.latex.preamble":
                '\n'.join([
                    '\\usepackage{amsmath}'
                    '\\usepackage[T1]{fontenc}', '\\usepackage{courier}',
                    '\\usepackage[variablett]{lmodern}',
                    '\\usepackage[LGRgreek]{mathastext}',
                    '\\renewcommand{\\rmdefault}{\\ttdefault}'
                ])
            })

    if style == 'nature':
        rcParams.update({
            "font.family": "sans-serif",
            # The default edge colors for scatter plots.
            "scatter.edgecolors": "black",
            "mathtext.fontset": "stixsans"
        })


def muRe_to_intenRe(muRe, zpt, pixel_size=0.259):
    """[summary]

    Args:
        muRe ([type]): [description]
        zpt ([type]): [description]

    Returns:
        [type]: [description]
    """
    intenRe = 10**((zpt - muRe) / 2.5) * pixel_size**2
    return intenRe

def sersic_bn(n):
    """ Accurate Sersic bn.

    Args:
        n (float): Sersic index

    Returns:
        bn: bn in the Sersic function.
    """

    bn = gammaincinv(2. * n, 0.5)

    return bn


def Sersic_intens(r, Ie, r_eff, n):
    """_summary_

    Args:
        r (_type_): _description_
        Ie (_type_): _description_
        r_eff (_type_): _description_
        n (_type_): _description_
    """
    bn = sersic_bn(n)

    intensity = Ie * np.exp(-bn * ((r / r_eff)**(1 / n) - 1))

    return intensity


def Exponential_intens(r, I0, rs):
    """_summary_

    Args:
        r (_type_): _description_
        I0 (_type_): _description_
        rs (_type_): _description_
    """

    intensity = I0 * np.exp(-r / rs)

    return intensity

# Total magnitudes, assuming that inputs are in units of
# counts/pixel and dimensions are in pixels

def TotalMagExp(I_0, h, ell, zeroPoint=0, magOut=True):
    """Calculate the total magnitude (or flux if magOut=False) for an
    2D exponential with [I_0, h] = params, where I_0 is in counts/pixel
    and h is in pixels.  Optionally, the ellipticity can be specified.
    """

    totalFlux = 2 * np.pi * I_0 * (h*h) * (1.0 - ell)
    if magOut:
        return (zeroPoint - 2.5 * np.log10(totalFlux))
    else:
        return totalFlux

def TotalMagSersic(n, I_e, r_e, ell, zeroPoint=0, magOut=True):
    """Calculate the total magnitude (or flux if magOut=False) for a
    2D Sersic function with [n, I_e, r_e] = params, where I_0 is in counts/pixel
    and h is in pixels.  Optionally, the ellipticity can be specified.
    """
    from scipy.special import gamma as Gamma
    
    bn = sersic_bn(n)
    bn2n = bn**(2*n)
    totalFlux = 2 * np.pi * n * np.exp(bn) * I_e * (r_e*r_e) * (1.0 - ell)
    totalFlux = totalFlux * Gamma(2*n) / bn2n
    if magOut:
        return (zeroPoint - 2.5 * np.log10(totalFlux))
    else:
        return totalFlux
    
def getBtoT(n, I_e, r_e, ell_bulge, I_0, h, ell_disk):
    flux_sersic_bulge = TotalMagSersic(n, I_e, r_e, ell_bulge, magOut=False)
    flux_exp_disk = TotalMagExp(I_0, h, ell_disk, magOut=False)
    
    BtoT = flux_sersic_bulge/(flux_exp_disk+flux_sersic_bulge)
    
    return BtoT

def nantozero(data):
    temp = []
    for i in range(len(data)):
        if np.isnan(data[i]):
            data[i] = 0
        temp.append(data[i])

    return np.array(temp)


def pospa(pa):
    if isinstance(pa, int) or isinstance(pa, float):
        temp = 0
        if pa < 0:
            temp = pa + 180
        else:
            temp = pa

    else:
        temp = []
        for i in range(len(pa)):
            if pa[i] < 0:
                temp.append(pa[i] + 180)
            else:
                temp.append(pa[i])
        temp = np.array(temp)

    return temp


def all_ninety_pa(pa):
    """This is for modifying the position angle to be in the range of -90-90 for IRAF.

    Args:
        pa (_type_): _description_

    Returns:
        _type_: _description_
    """
    temp = 0
    if pa > 90:
        temp = pa - 180
    else:
        temp = pa

    return temp


def bright_to_mag(intens, zpt0, texp, pixel_size):
    # for CGS survey, texp = 1s, A = 0.259*0.259
    texp = texp
    A = pixel_size**2
    return -2.5 * np.log10(intens / (texp * A)) + zpt0


def mag2bright(mag, zpt0, texp, pixel_size):
    A = pixel_size**2

    bright = texp * A * 10**((mag - zpt0) / (-2.5))

    return bright


def bright_to_mag_DESI(intens):
    """transform the flux to magnitude. 

    Args:
        intens (numpy array): the instensity profiles. #! The unit should be nanomaggie/arecsec2.
        zpt0 (float): zero point of the DESI AB magnitude system. here is 22.5 for 1 naonomaggie.

    Returns:
        mu: surface brightness profiles.
    """
    mu = -2.5 * np.log10(intens) + 22.5

    return mu


def inten_to_mag(intens, zpt0):
    '''
    This function is for calculating the magnitude of total intensity.
    '''
    return -2.5 * np.log10(intens) + zpt0


def asymmetry_propagate_err_mu(inten, err):
    detup_residual = 2.5 * np.log10((inten + err) / inten)
    detdown_residual = 2.5 * np.log10(inten / (inten - err))

    return detup_residual, detdown_residual


def symmetry_propagate_err_mu(intens, intens_err):
    """Get the symmetry magnitude error based on the error propagation formula.

    Args:
        intens (numpy array): the isophotal intensity.
        intens_err (numpy array): the isophotal intensity error.

    Returns:
        mu_err: the symmetry magnitude error.
    """

    mu_err = np.array(2.5 / np.log(10) * intens_err / intens)

    return mu_err


def subtract_sky(input_file, sky_value, subtract_sky_file):
    """
    This function is to subtract the sky value form the original image.
    It is to be modifed, use the easy way to write just the data into the file. Then we do not need consider the header.
    """

    Remove_file(subtract_sky_file)

    hdul = fits.open(input_file)
    data_list = hdul[0].data

    data_list -= sky_value

    hdul.writeto(subtract_sky_file)


def subtract_sky_woheader(input_file, sky_value, subtract_sky_file):
    '''
    This function is to subtract the sky value form the original image.
    It is to be modifed, use the easy way to write just the data into the file. Then we do not need consider the header.

    This function is especially for those fits whose header has some conflicts.
    '''

    Remove_file(subtract_sky_file)

    hdul = fits.open(input_file)
    header = hdul[0].header
    data_list = hdul[0].data

    data_list -= sky_value

    easy_saveData_Tofits(data_list, header, savefile=subtract_sky_file)


def removeellipseIndef(arr):
    # let the indef equals NaN
    arr[arr == 'INDEF'] = np.nan

    # convert the str array into float array

    arr_new = [float(arr[i]) for i in range(len(arr))]

    return np.array(arr_new)


def imageMax(image_data_temp, mask_data_temp):
    # combine the image and mask
    image_data_temp[mask_data_temp > 0] = np.nan
    image_value_max = np.nanmax(image_data_temp)

    return image_value_max


def ellipseGetGrowthCurve(ellipOut, useTflux=False):
    """
    Extract growth curve from Ellipse output.
    Parameters: ellipOut
    Parameters: useTflux, if use the cog of iraf or not.
    """
    if not useTflux:
        # The area in unit of pixels covered by an elliptical isophote
        ell = removeellipseIndef(ellipOut['ell'])
        ellArea = np.pi * ((ellipOut['sma']**2.0) * (1.0 - ell))
        # The total flux inside the "ring"
        intensUse = ellipOut['intens']
        isoFlux = np.append(ellArea[0],
                            [ellArea[1:] - ellArea[:-1]]) * ellipOut['intens']
        # Get the growth Curve
        curveOfGrowth = list(
            map(lambda x: np.nansum(isoFlux[0:x + 1]),
                range(isoFlux.shape[0])))
    else:
        curveOfGrowth = ellipOut['tflux_e']

    indexMax = np.nanargmax(removeellipseIndef(curveOfGrowth))
    maxIsoSma = ellipOut['sma'][indexMax]
    maxIsoFlux = curveOfGrowth[indexMax]

    return np.asarray(curveOfGrowth), maxIsoSma, maxIsoFlux


def GrowthCurve(sma, ellip, isoInten):
    """THis function is to derive the curve of growth by integrating the isophotes.

    Args:
        sma (numpy array): The radial radius array. The unit should be arcsec? #!for CGS it should be pixel, it depends on the unit of the intensity. The default unit of CGS isophote is ADUs/pixel. 
        ellip (float): the fixed ellipticity.
        isoInten (numpy array): the intensity array at each radius.

    Returns:
        [type]: [description]
    """
    ellArea = np.pi * ((sma**2.0) * (1.0 - ellip))
    isoFlux = np.append(ellArea[0], [ellArea[1:] - ellArea[:-1]]) * isoInten
    curveOfGrowth = list(
        map(lambda x: np.nansum(isoFlux[0:x + 1]), range(isoFlux.shape[0])))

    indexMax = np.argmax(curveOfGrowth)
    maxIsoSma = sma[indexMax]
    maxIsoFlux = curveOfGrowth[indexMax]

    return np.asarray(curveOfGrowth), maxIsoSma, maxIsoFlux


def GrowthCurve_trapz(R, I, axisratio):
    """
    Integrate a flux density profile from Autoprof writter by Connor Stone.

    R: semi-major axis length (arcsec)
    I: flux density (flux/arcsec^2)
    axisratio: b/a profile
    """

    S = np.zeros(len(R))
    S[0] = I[0] * np.pi * axisratio[0] * (R[0] ** 2)
    for i in range(1, len(R)):
        S[i] = (
            np.trapz(2 * np.pi * I[: i + 1] * R[: i + 1] * axisratio[: i + 1], R[: i + 1])
            + S[0]
        )
    return S


def get_Rpercent(sma, cog, maxFlux, percent):
    cog_percent = maxFlux * percent

    f = interp1d(cog, sma)

    Rpercent = f(cog_percent)

    return Rpercent


def get_Rmag(sma, mu, mag_use):
    f = interp1d(mu, sma)

    Rnum = f(mag_use)

    return Rnum


def get_Ie(sma, intens, re):
    f = interp1d(sma, intens)

    Ie = f(re)

    return Ie


def correct_pa_profile(ellipse_output, delta_pa=75.0):
    """
    Correct the position angle for large jump.
    Parameters
    ----------
    ellipse_output: astropy.table
        Output table summarizing the result from `ellipse`.
    pa_col: string, optional
        Name of the position angle column. Default: pa
    delta_pa: float, optional
        Largest PA difference allowed for two adjacent radial bins. Default=75.
    Return
    ------
    ellipse_output with updated position angle column.
    """
    pa = ellipse_output['pa'].astype('float')

    for i in range(1, len(pa)):
        if (pa[i] - pa[i - 1]) >= delta_pa:
            pa[i] -= 180.0
        elif pa[i] - pa[i - 1] <= (-1.0 * delta_pa):
            pa[i] += 180.0

    ellipse_output['pa'] = pa

    return ellipse_output


def correct_pa_single(pa_arr, delta_pa=75.0):
    """
    Correct the position angle for large jump.
    Parameters
    ----------
    pa_arr: np.array
        PA array to be fixed
    delta_pa: float, optional
        Largest PA difference allowed for two adjacent radial bins. Default=75.
    Return
    ------
    ellipse_output with updated position angle column.
    """
    pa = np.array(pa_arr)

    for i in range(1, len(pa)):
        if (pa[i] - pa[i - 1]) >= delta_pa:
            pa[i] -= 180.0
        elif pa[i] - pa[i - 1] <= (-1.0 * delta_pa):
            pa[i] += 180.0

    return pa


def normalize_angle(num, lower=0, upper=360, b=False):
    """Normalize number to range [lower, upper) or [lower, upper].
    Parameters
    ----------
    num : float
        The number to be normalized.
    lower : int
        Lower limit of range. Default is 0.
    upper : int
        Upper limit of range. Default is 360.
    b : bool
        Type of normalization. Default is False. See notes.
    Returns
    -------
    n : float
        A number in the range [lower, upper) or [lower, upper].
    """
    from math import ceil, floor

    # abs(num + upper) and abs(num - lower) are needed, instead of
    # abs(num), since the lower and upper limits need not be 0. We need
    # to add half size of the range, so that the final result is lower +
    # <value> or upper - <value>, respectively.
    res = num
    if not b:
        if lower >= upper:
            raise ValueError("Invalid lower and upper limits: (%s, %s)" %
                             (lower, upper))

        res = num
        if num > upper or num == lower:
            num = lower + abs(num + upper) % (abs(lower) + abs(upper))
        if num < lower or num == upper:
            num = upper - abs(num - lower) % (abs(lower) + abs(upper))

        res = lower if num == upper else num
    else:
        total_length = abs(lower) + abs(upper)
        if num < -total_length:
            num += ceil(num / (-2 * total_length)) * 2 * total_length
        if num > total_length:
            num -= floor(num / (2 * total_length)) * 2 * total_length
        if num > upper:
            num = total_length - num
        if num < lower:
            num = -total_length - num

        res = num

    res *= 1.0  # Make all numbers float, to be consistent

    return res


def readEllipse(outDat, zpt0, sky_err, pixel_size=0.259, sky_value=0, texp=1):
    # read the data
    ellipse_data = Table.read(outDat, format='ascii.no_header')
    ellipse_data.rename_column('col1', 'sma')
    ellipse_data.rename_column('col2', 'intens')
    ellipse_data.rename_column('col3', 'int_err')
    ellipse_data.rename_column('col4', 'pix_var')
    ellipse_data.rename_column('col5', 'rms')
    ellipse_data.rename_column('col6', 'ell')
    ellipse_data.rename_column('col7', 'ell_err')
    ellipse_data.rename_column('col8', 'pa')
    ellipse_data.rename_column('col9', 'pa_err')
    ellipse_data.rename_column('col10', 'x0')
    ellipse_data.rename_column('col11', 'x0_err')
    ellipse_data.rename_column('col12', 'y0')
    ellipse_data.rename_column('col13', 'y0_err')
    ellipse_data.rename_column('col14', 'grad')
    ellipse_data.rename_column('col15', 'grad_err')
    ellipse_data.rename_column('col16', 'grad_r_err')
    ellipse_data.rename_column('col17', 'rsma')
    ellipse_data.rename_column('col18', 'mag')
    ellipse_data.rename_column('col19', 'mag_lerr')
    ellipse_data.rename_column('col20', 'mag_uerr')
    ellipse_data.rename_column('col21', 'tflux_e')
    ellipse_data.rename_column('col22', 'tflux_c')
    ellipse_data.rename_column('col23', 'tmag_e')
    ellipse_data.rename_column('col24', 'tmag_c')
    ellipse_data.rename_column('col25', 'npix_e')
    ellipse_data.rename_column('col26', 'npix_c')
    ellipse_data.rename_column('col27', 'a3')
    ellipse_data.rename_column('col28', 'a3_err')
    ellipse_data.rename_column('col29', 'b3')
    ellipse_data.rename_column('col30', 'b3_err')
    ellipse_data.rename_column('col31', 'a4')
    ellipse_data.rename_column('col32', 'a4_err')
    ellipse_data.rename_column('col33', 'b4')
    ellipse_data.rename_column('col34', 'b4_err')
    ellipse_data.rename_column('col35', 'ndata')
    ellipse_data.rename_column('col36', 'nflag')
    ellipse_data.rename_column('col37', 'niter')
    ellipse_data.rename_column('col38', 'stop')
    ellipse_data.rename_column('col39', 'a_big')
    ellipse_data.rename_column('col40', 'sarea')

    # Normalize the PA
    dPA = 75.0
    ellipse_data['pa'] = removeellipseIndef(ellipse_data['pa'])
    pa_correct = correct_pa_single(ellipse_data['pa'], delta_pa=dPA)

    column_pa_norm = Column(name='pa_norm',
                            data=np.array([
                                normalize_angle(pa,
                                                lower=0,
                                                upper=180,
                                                b=False) for pa in pa_correct
                            ]))
    ellipse_data.add_column(column_pa_norm)

    # remove the indef
    intens = ellipse_data['intens']
    intens_removeindef = np.array(removeellipseIndef(intens))
    intens_err = ellipse_data['int_err']
    intens_err_removeindef = removeellipseIndef(intens_err)

    ellipse_data['ell'] = removeellipseIndef(ellipse_data['ell'])
    ellipse_data['pa'] = removeellipseIndef(ellipse_data['pa'])
    ellipse_data['ell_err'] = removeellipseIndef(ellipse_data['ell_err'])
    ellipse_data['pa_err'] = removeellipseIndef(ellipse_data['pa_err'])

    # calculate the magnitude.
    intens_err_removeindef_sky = np.sqrt(
        np.array(intens_err_removeindef)**2 + sky_err**2)

    if sky_value:
        ellipse_data['intens'] = intens_removeindef - sky_value
        mu = bright_to_mag(intens_removeindef - sky_value, zpt0, texp,
                           pixel_size)
        mu_err = symmetry_propagate_err_mu(intens_removeindef - sky_value,
                                           intens_err_removeindef_sky)
    else:
        ellipse_data['intens'] = intens_removeindef
        mu = bright_to_mag(intens_removeindef, zpt0, texp, pixel_size)
        mu_err = symmetry_propagate_err_mu(intens_removeindef,
                                           intens_err_removeindef_sky)

    ellipse_data.add_column(Column(name='mu', data=mu))
    ellipse_data.add_column(Column(name='mu_err', data=mu_err))

    return ellipse_data


def readGalfitInput(input_file):
    with open(input_file) as f:
        input_data = f.read()

    mue = re.search('(?<=3\)\s).*(?=\s[0-9])', input_data)[0]
    Re = re.search('(?<=4\)\s).*(?=\s[0-9])', input_data)[0]
    n = re.search('(?<=5\)\s).*(?=\s[0-9])', input_data)[0]

    sky_value_t = re.search('(?<=1\)\s).*(?=#\s\sSky)', input_data)[0]
    sky_value = re.search('.*(?=\s[0-9])', sky_value_t)[0]

    return np.array([mue, Re, n, sky_value])


def numpy_weighted_mean(data, weights=None):
    """Calculate the weighted mean of an array/list using numpy."""
    weights = np.array(weights).flatten() / float(sum(weights))

    return np.dot(np.array(data), weights)


def ellipseGetAvgGeometry(ellipseOut, outRad, minSma=2.0):
    """Get the Average Q and PA."""
    # tfluxE = ellipseOut['tflux_e']
    # ringFlux = np.append(tfluxE[0], [tfluxE[1:] - tfluxE[:-1]])

    tfluxE = removeellipseIndef(ellipseOut['tflux_e'])
    ringFlux = np.append(tfluxE[0], [tfluxE[1:] - tfluxE[:-1]])

    try:
        eUse = ellipseOut['ell'][(ellipseOut['sma'] <= outRad)
                                 & (ellipseOut['sma'] >= minSma) &
                                 (np.isfinite(ellipseOut['ell_err'])) &
                                 (np.isfinite(ellipseOut['pa_err']))]
        pUse = ellipseOut['pa_norm'][(ellipseOut['sma'] <= outRad)
                                     & (ellipseOut['sma'] >= minSma) &
                                     (np.isfinite(ellipseOut['ell_err'])) &
                                     (np.isfinite(ellipseOut['pa_err']))]
        fUse = ringFlux[(ellipseOut['sma'] <= outRad)
                        & (ellipseOut['sma'] >= minSma) &
                        (np.isfinite(ellipseOut['ell_err'])) &
                        (np.isfinite(ellipseOut['pa_err']))]
    except Exception:
        try:
            eUse = ellipseOut['ell'][(ellipseOut['sma'] <= outRad)
                                     & (ellipseOut['sma'] >= 0.5) &
                                     (np.isfinite(ellipseOut['ell_err'])) &
                                     (np.isfinite(ellipseOut['pa_err']))]
            pUse = ellipseOut['pa_norm'][(ellipseOut['sma'] <= outRad)
                                         & (ellipseOut['sma'] >= 0.5) &
                                         (np.isfinite(ellipseOut['ell_err'])) &
                                         (np.isfinite(ellipseOut['pa_err']))]
            fUse = ringFlux[(ellipseOut['sma'] <= outRad)
                            & (ellipseOut['sma'] >= 0.5) &
                            (np.isfinite(ellipseOut['ell_err'])) &
                            (np.isfinite(ellipseOut['pa_err']))]
        except Exception:
            eUse = ellipseOut['ell'][(ellipseOut['sma'] <= outRad)
                                     & (ellipseOut['sma'] >= 0.5)]
            pUse = ellipseOut['pa_norm'][(ellipseOut['sma'] <= outRad)
                                         & (ellipseOut['sma'] >= 0.5)]
            fUse = ringFlux[(ellipseOut['sma'] <= outRad)
                            & (ellipseOut['sma'] >= 0.5)]

    avgQ = 1.0 - numpy_weighted_mean(eUse, weights=fUse)
    avgPA = numpy_weighted_mean(pUse, weights=fUse)

    return avgQ, avgPA


def ellipseGetAvgGeometry_CoG(ellipseOut, outRad, minSma=2.0):
    """Get the Average Q and PA."""
    # tfluxE = ellipseOut['tflux_e']
    # ringFlux = np.append(tfluxE[0], [tfluxE[1:] - tfluxE[:-1]])

    # tfluxE = removeellipseIndef(ellipseOut['tflux_e'])
    tflux, maxIsoSma, maxIsoFlux = GrowthCurve(ellipseOut['sma'],
                                               ellipseOut['ell'],
                                               ellipseOut['intens'])
    ringFlux = np.append(tflux[0], [tflux[1:] - tflux[:-1]])

    try:
        eUse = ellipseOut['ell'][(ellipseOut['sma'] <= outRad)
                                 & (ellipseOut['sma'] >= minSma) &
                                 (np.isfinite(ellipseOut['ell_err'])) &
                                 (np.isfinite(ellipseOut['pa_err']))]
        pUse = ellipseOut['pa_norm'][(ellipseOut['sma'] <= outRad)
                                     & (ellipseOut['sma'] >= minSma) &
                                     (np.isfinite(ellipseOut['ell_err'])) &
                                     (np.isfinite(ellipseOut['pa_err']))]
        fUse = ringFlux[(ellipseOut['sma'] <= outRad)
                        & (ellipseOut['sma'] >= minSma) &
                        (np.isfinite(ellipseOut['ell_err'])) &
                        (np.isfinite(ellipseOut['pa_err']))]
    except Exception:
        try:
            eUse = ellipseOut['ell'][(ellipseOut['sma'] <= outRad)
                                     & (ellipseOut['sma'] >= 0.5) &
                                     (np.isfinite(ellipseOut['ell_err'])) &
                                     (np.isfinite(ellipseOut['pa_err']))]
            pUse = ellipseOut['pa_norm'][(ellipseOut['sma'] <= outRad)
                                         & (ellipseOut['sma'] >= 0.5) &
                                         (np.isfinite(ellipseOut['ell_err'])) &
                                         (np.isfinite(ellipseOut['pa_err']))]
            fUse = ringFlux[(ellipseOut['sma'] <= outRad)
                            & (ellipseOut['sma'] >= 0.5) &
                            (np.isfinite(ellipseOut['ell_err'])) &
                            (np.isfinite(ellipseOut['pa_err']))]
        except Exception:
            eUse = ellipseOut['ell'][(ellipseOut['sma'] <= outRad)
                                     & (ellipseOut['sma'] >= 0.5)]
            pUse = ellipseOut['pa_norm'][(ellipseOut['sma'] <= outRad)
                                         & (ellipseOut['sma'] >= 0.5)]
            fUse = ringFlux[(ellipseOut['sma'] <= outRad)
                            & (ellipseOut['sma'] >= 0.5)]

    avgQ = 1.0 - numpy_weighted_mean(eUse, weights=fUse)
    avgPA = numpy_weighted_mean(pUse, weights=fUse)

    return avgQ, avgPA

def Remove_file(file):
    if os.path.exists(file):
        os.remove(file)

def calculate_sky_stats(image_data, box_size = 20, nan_frac = 0.5, show_fig = False):
    from astropy.nddata import block_reduce

    from Taiji.imtools import extract_obj
    
    sep_res = extract_obj(image_data, convolve=True, conv_radius = 3, convolve_kernel='gaussian')
    mask_data_all = sep_res[1]

    mask_all_copy = copy.deepcopy(mask_data_all)
    
    if show_fig:
        _ = display_single(mask_all_copy.astype('float'))
        
    mask_all_totalnumber = block_reduce(np.ones_like(image_data), box_size, np.sum)
    mask_all_binnumber = block_reduce(mask_all_copy, box_size, np.sum)
    index_percentile = (mask_all_binnumber/mask_all_totalnumber)>=nan_frac

    img_copy = copy.deepcopy(image_data)

    img_copy[mask_data_all==1] = np.nan

    img_bin = block_reduce(img_copy, box_size, np.nanmedian)
    img_bin[index_percentile] = np.nan

    img_bin_std = block_reduce(img_copy, box_size, np.nanstd)
    img_bin_std[index_percentile] = np.nan

    sky_rms = np.sqrt(np.nansum(img_bin**2)/len(img_bin[~np.isnan(img_bin)])) # sky level uncertainty, refer to Gao and ho 2017 and this should be the DSKY in S4G
    sky_value = np.nanmedian(img_bin)
    sky_std = np.nanstd(img_bin)
    sky_local_rms = np.nanmedian(img_bin_std) # RMS in S4G
    
    sky_stats = {'sky value': sky_value, 'sky std (box bin)': sky_std, 'sky uncertainty': sky_rms, 'sky local rms': sky_local_rms}
    
    return sky_stats

def plot_ellipse(ellipse_data, outer_limit, pixel_size=0.259):
    '''
    This function is to illustrate the sbp easily. It is to be developed into a more genaral version. Like Song and Jiaxuan do.
    '''
    sma = ellipse_data['sma']
    intens = ellipse_data['intens']
    mu = ellipse_data['mu']
    mu_err = ellipse_data['mu_err']

    index = intens > outer_limit
    sma_sky = sma[index]
    mu_sky = mu[index]
    mu_err_sky = mu_err[index]

    plt.plot(sma_sky * pixel_size, mu_sky)
    plt.fill_between(sma_sky * pixel_size,
                     mu_sky - mu_err_sky,
                     mu_sky + mu_err_sky,
                     alpha=0.2)
    plt.ylim(np.min(mu_sky) - 0.2, np.max(mu_sky) + 0.2)
    plt.gca().invert_yaxis()
    plt.ylabel(r'$\mu_R\ (\mathrm{mag\ arcsec^{-2}})$')
    plt.xlabel(r'$r\,(\mathrm{arcsec})$')


def getOuterBound(ellipse_data, sky_err, zpt0, alter=0.2):
    sma = ellipse_data['sma']
    intens = ellipse_data['intens']
    mu = ellipse_data['mu']
    mu_err = ellipse_data['mu_err']

    mu_err_justsky = symmetry_propagate_err_mu(intens, sky_err)

    index = mu_err_justsky <= alter

    return sma[index][-1]


def getBound(sma, intens, int_err, zpt0, pixel_size=0.259, texp=1, alter=0.2):
    mu = bright_to_mag(intens=intens,
                       zpt0=zpt0,
                       texp=texp,
                       pixel_size=pixel_size)
    mu_err = symmetry_propagate_err_mu(intens=intens, intens_err=int_err)

    index = np.abs(mu_err) <= alter

    return np.array([sma[index][0], sma[index][-1]], dtype=float)


def plot_x0(ax,
            sma,
            x0,
            x0_err,
            pixel_size=0.259,
            plot_style='fill',
            color='k',
            ylimin=None,
            ylimax=None,
            xlimin=None,
            xlimax=None,
            label='Center'):
    '''
    This function is a templete to plot the center x0 profile.
    '''

    if plot_style == 'errorbar':
        ax.errorbar(sma * pixel_size,
                    x0,
                    yerr=x0_err,
                    fmt='o',
                    markersize=3,
                    color=color,
                    capsize=3,
                    elinewidth=0.7,
                    label=label)

    elif plot_style == 'fill':
        ax.plot(sma * pixel_size, x0, color=color, lw=3, label=label)
        ax.fill_between(sma * pixel_size,
                        x0 + x0_err,
                        x0 - x0_err,
                        color=color,
                        alpha=0.5)

    if ylimax:
        ax.set_ylim(ylimin, ylimax)
    else:
        ax.set_ylim(np.nanmin(x0) - 5, np.nanmax(x0) + 5)

    if xlimax:
        ax.set_xlim(xlimin, xlimax)

    ax.set_ylabel(r'Center')
    ax.set_xlabel(r'$r\,(\mathrm{arcsec})$')
    ax.legend()


def plot_ellip(ax,
               sma,
               ellip,
               ellip_err,
               pixel_size=0.259,
               plot_style='fill',
               color='k',
               ylimin=None,
               ylimax=None,
               xlimin=None,
               xlimax=None,
               **kwargs):
    '''
    This function is a templete to plot the ellipticity profile.
    '''

    if plot_style == 'errorbar':
        ax.errorbar(sma * pixel_size,
                    ellip,
                    yerr=ellip_err,
                    fmt='o',
                    markersize=3,
                    color=color,
                    capsize=3,
                    elinewidth=0.7,
                    **kwargs)

    elif plot_style == 'fill':
        ax.plot(sma * pixel_size, ellip, color=color, **kwargs)
        ax.fill_between(sma * pixel_size,
                        ellip + ellip_err,
                        ellip - ellip_err,
                        color=color,
                        alpha=0.5)

    elif plot_style == 'line':
        ax.plot(sma * pixel_size, ellip, color=color, **kwargs)

    if ylimax:
        ax.set_ylim(ylimin, ylimax)
    else:
        ax.set_ylim(np.nanmin(ellip) - 0.05, np.nanmax(ellip) + 0.05)

    if xlimax:
        ax.set_xlim(xlimin, xlimax)

    ax.set_ylabel(r'Ellipticity')
    ax.set_xlabel(r'$r\,(\mathrm{arcsec})$')
    ax.legend()


def plot_axisRatio(ax,
                   sma,
                   axisratio,
                   axisratio_err,
                   pixel_size=0.259,
                   plot_style='fill',
                   color='k',
                   ylimin=None,
                   ylimax=None,
                   xlimin=None,
                   xlimax=None,
                   **kwargs):
    '''
    This function is a templete to plot the ellipticity profile.
    '''

    if plot_style == 'errorbar':
        ax.errorbar(sma * pixel_size,
                    axisratio,
                    yerr=axisratio_err,
                    fmt='o',
                    markersize=3,
                    color=color,
                    capsize=3,
                    elinewidth=0.7,
                    **kwargs)

    elif plot_style == 'fill':
        ax.plot(sma * pixel_size, axisratio, color=color, **kwargs)
        ax.fill_between(sma * pixel_size,
                        axisratio + axisratio_err,
                        axisratio - axisratio_err,
                        color=color,
                        alpha=0.5)

    elif plot_style == 'line':
        ax.plot(sma * pixel_size, axisratio, color=color, **kwargs)

    if ylimax:
        ax.set_ylim(ylimin, ylimax)
    else:
        ax.set_ylim(np.nanmin(axisratio) - 0.05, np.nanmax(axisratio) + 0.05)

    if xlimax:
        ax.set_xlim(xlimin, xlimax)

    ax.set_ylabel(r'$b/a$')
    ax.set_xlabel(r'$r\,(\mathrm{arcsec})$')
    # ax.legend()


def plot_pa(ax,
            sma,
            pa,
            pa_err,
            pixel_size=0.259,
            plot_style='fill',
            color='k',
            ylimin=None,
            ylimax=None,
            xlimin=None,
            xlimax=None,
            **kwargs):
    '''
    This function is a templete to plot the PA profile.
    '''

    if plot_style == 'errorbar':
        ax.errorbar(sma * pixel_size,
                    pa,
                    yerr=pa_err,
                    fmt='o',
                    markersize=3,
                    color=color,
                    capsize=3,
                    elinewidth=0.7,
                    **kwargs)

    elif plot_style == 'fill':
        ax.plot(sma * pixel_size, pa, color=color, **kwargs)
        ax.fill_between(sma * pixel_size,
                        pa + pa_err,
                        pa - pa_err,
                        color=color,
                        alpha=0.5)

    elif plot_style == 'line':
        ax.plot(sma * pixel_size, pa, color=color, **kwargs)

    if ylimax:
        ax.set_ylim(ylimin, ylimax)
    else:
        ax.set_ylim(np.nanmin(pa) - 5, np.nanmax(pa) + 5)

    if xlimax:
        ax.set_xlim(xlimin, xlimax)

    #     else:
    #         plt.xlim(sma[-1]*0.02*(-1)*pixel_size, (sma[-1]+sma[-1]*0.02)*pixel_size)

    ax.set_ylabel(r'PA\,(deg)')
    ax.set_xlabel(r'$r\,(\mathrm{arcsec})$')
    # ax.legend()


def plot_SBP(ax,
             sma,
             mu,
             mu_err,
             pixel_size=0.259,
             plot_style='fill',
             color='k',
             ylimin=None,
             ylimax=None,
             xlimin=None,
             xlimax=None,
             **kwargs):
    '''
    This function is a templete to plot the SB profile.
    '''

    if plot_style == 'errorbar':
        ax.errorbar(sma * pixel_size,
                    mu,
                    yerr=mu_err,
                    fmt='o',
                    markersize=3,
                    color=color,
                    capsize=3,
                    elinewidth=0.7,
                    **kwargs)

    elif plot_style == 'fill':
        ax.plot(sma * pixel_size, mu, color=color, **kwargs)
        ax.fill_between(sma * pixel_size,
                        mu + mu_err,
                        mu - mu_err,
                        color=color,
                        alpha=0.5)

    elif plot_style == 'line':
        ax.plot(sma * pixel_size, mu, color=color, **kwargs)

    if ylimax:
        ax.set_ylim(ylimin, ylimax)
    else:
        ax.set_ylim(np.nanmin(mu) - 0.5, np.nanmax(mu) + 0.5)

    if xlimax:
        ax.set_xlim(xlimin, xlimax)

    ax.set_ylabel(r'$\mu_R\ (\mathrm{mag\ arcsec^{-2}})$')
    ax.set_xlabel(r'$r\,(\mathrm{arcsec})$')
    ax.legend()
    plt.gca().invert_yaxis()

def plot_completeSBP_firststep(sma,
                               x0,
                               x0_err,
                               y0,
                               y0_err,
                               ell,
                               ell_err,
                               pa,
                               pa_err,
                               mu,
                               mu_err,
                               intens_subbkg,
                               sky_err,
                               pixel_size=0.259,
                               plot_style='fill',
                               color='k',
                               xlimin=None,
                               xlimax=None,
                               ylimin_e=None,
                               ylimax_e=None,
                               ylimin_pa=None,
                               ylimax_pa=None,
                               ylimin_mu=None,
                               ylimax_mu=None,
                               save_file=''):
    '''
    This function is to plot the standard and basic surface brightness profiles including sbp, ell, and pa panels.
    '''
    fig = plt.figure(figsize=(10, 14.5))
    fig.subplots_adjust(left=1, right=2, top=1, bottom=0, wspace=0, hspace=0)
    gs = GridSpec(ncols=1, nrows=29, figure=fig)

    if not xlimin:
        deltaN = 0.05
        index_above_sigma = intens_subbkg > sky_err
        len_xlim = len(sma[index_above_sigma]) * pixel_size

        xlimin = -deltaN * len_xlim

        xlimax = (sma[index_above_sigma][-1]) * pixel_size + deltaN * len_xlim

    ax1 = fig.add_subplot(gs[:5, 0])
    plot_x0(ax1,
            sma[index_above_sigma],
            x0[index_above_sigma],
            x0_err[index_above_sigma],
            pixel_size=0.259,
            plot_style='fill',
            color='k',
            ylimin=None,
            ylimax=None,
            xlimin=xlimin,
            xlimax=xlimax,
            label='x0')
    plot_x0(ax1,
            sma[index_above_sigma],
            y0[index_above_sigma],
            y0_err[index_above_sigma],
            pixel_size=0.259,
            plot_style='errorbar',
            color='k',
            ylimin=None,
            ylimax=None,
            xlimin=xlimin,
            xlimax=xlimax,
            label='y0')

    ax2 = fig.add_subplot(gs[5:10, 0])
    plot_ellip(ax2,
               sma[index_above_sigma],
               ell[index_above_sigma],
               ell_err[index_above_sigma],
               pixel_size=pixel_size,
               plot_style=plot_style,
               color=color,
               ylimin=ylimin_e,
               ylimax=ylimax_e,
               xlimin=xlimin,
               xlimax=xlimax)

    ax3 = fig.add_subplot(gs[10:15, 0])
    plot_pa(ax3,
            sma[index_above_sigma],
            pa[index_above_sigma],
            pa_err[index_above_sigma],
            pixel_size=pixel_size,
            plot_style=plot_style,
            color=color,
            ylimin=ylimin_pa,
            ylimax=ylimax_pa,
            xlimin=xlimin,
            xlimax=xlimax)

    ax4 = fig.add_subplot(gs[15:, 0])
    plot_SBP(ax4,
             sma[index_above_sigma],
             mu[index_above_sigma],
             mu_err[index_above_sigma],
             pixel_size=pixel_size,
             plot_style=plot_style,
             color=color,
             ylimin=ylimin_mu,
             ylimax=ylimax_mu,
             xlimin=xlimin,
             xlimax=xlimax)

    if save_file:
        plt.savefig(save_file, dpi=200, bbox_inches='tight')
    # plt.show()


def plot_completeSBP(sma,
                     ell,
                     ell_err,
                     pa,
                     pa_err,
                     mu,
                     mu_err,
                     intens_subbkg,
                     sky_err,
                     pixel_size=0.259,
                     plot_style='fill',
                     color='k',
                     xlimin=None,
                     xlimax=None,
                     ylimin_e=None,
                     ylimax_e=None,
                     ylimin_pa=None,
                     ylimax_pa=None,
                     ylimin_mu=None,
                     ylimax_mu=None,
                     save_file=''):
    '''
    This function is to plot the standard and basic surface brightness profiles including sbp, ell, and pa panels.
    '''
    fig = plt.figure(figsize=(10, 12))
    fig.subplots_adjust(left=1, right=2, top=1, bottom=0, wspace=0, hspace=0)
    gs = GridSpec(ncols=1, nrows=24, figure=fig)

    if not xlimin:
        deltaN = 0.05
        index_above_sigma = intens_subbkg > sky_err
        len_xlim = len(sma[index_above_sigma]) * pixel_size

        xlimin = -deltaN * len_xlim

        xlimax = (sma[index_above_sigma][-1]) * pixel_size + deltaN * len_xlim

    ax1 = fig.add_subplot(gs[:5, 0])
    plot_ellip(ax1,
               sma[index_above_sigma],
               ell[index_above_sigma],
               ell_err[index_above_sigma],
               pixel_size=pixel_size,
               plot_style=plot_style,
               color=color,
               ylimin=ylimin_e,
               ylimax=ylimax_e,
               xlimin=xlimin,
               xlimax=xlimax)

    ax2 = fig.add_subplot(gs[5:10, 0])
    plot_pa(ax2,
            sma[index_above_sigma],
            pa[index_above_sigma],
            pa_err[index_above_sigma],
            pixel_size=pixel_size,
            plot_style=plot_style,
            color=color,
            ylimin=ylimin_pa,
            ylimax=ylimax_pa,
            xlimin=xlimin,
            xlimax=xlimax)

    ax3 = fig.add_subplot(gs[10:, 0])
    plot_SBP(ax3,
             sma[index_above_sigma],
             mu[index_above_sigma],
             mu_err[index_above_sigma],
             pixel_size=pixel_size,
             plot_style=plot_style,
             color=color,
             ylimin=ylimin_mu,
             ylimax=ylimax_mu,
             xlimin=xlimin,
             xlimax=xlimax)

    if save_file:
        plt.savefig(save_file, dpi=200, bbox_inches='tight')
    # plt.show()


def random_cmap(ncolors=256, background_color='white'):
    """Random color maps.
    Generate a matplotlib colormap consisting of random (muted) colors.
    A random colormap is very useful for plotting segmentation images.
    Parameters
    ----------
    ncolors : int, optional
        The number of colors in the colormap.  The default is 256.
    random_state : int or `~numpy.random.RandomState`, optional
        The pseudo-random number generator state used for random
        sampling.  Separate function calls with the same
        ``random_state`` will generate the same colormap.
    Returns
    -------
    cmap : `matplotlib.colors.Colormap`
        The matplotlib colormap with random colors.
    Notes
    -----
    Based on: colormaps.py in photutils
    """
    prng = np.random.mtrand._rand

    h = prng.uniform(low=0.0, high=1.0, size=ncolors)
    s = prng.uniform(low=0.2, high=0.7, size=ncolors)
    v = prng.uniform(low=0.5, high=1.0, size=ncolors)

    hsv = np.dstack((h, s, v))
    rgb = np.squeeze(colors.hsv_to_rgb(hsv))

    if background_color is not None:
        if background_color not in colors.cnames:
            raise ValueError('"{0}" is not a valid background color '
                             'name'.format(background_color))
        rgb[0] = colors.hex2color(colors.cnames[background_color])

    return colors.ListedColormap(rgb)


def LSBImage(ax,
             dat,
             noise,
             pixel_size=0.168,
             bar_length=50,
             box_alpha=1,
             **kwargs):
    # plt.figure(figsize=(6, 6))
    ax.imshow(dat,
              origin="lower",
              cmap="Greys",
              norm=ImageNormalize(stretch=HistEqStretch(dat[dat <= 3 * noise]),
                                  clip=False,
                                  vmax=3 * noise,
                                  vmin=np.min(dat)),
              aspect='auto',
              **kwargs)
    my_cmap = copy.copy(cm.Greys_r)
    my_cmap.set_under("k", alpha=0)

    ax.imshow(np.ma.masked_where(dat < 3 * noise, dat),
              origin="lower",
              cmap=my_cmap,
              norm=ImageNormalize(stretch=LogStretch(), clip=False),
              clim=[3 * noise, None],
              interpolation='none',
              aspect='auto',
              **kwargs)

    scalebar = ScaleBar(pixel_size,
                        "''",
                        dimension=ANGLE,
                        color='black',
                        box_alpha=box_alpha,
                        font_properties={'size': 15},
                        location='lower left',
                        length_fraction=pixel_size,
                        fixed_value=bar_length)
    ax.add_artist(scalebar)

    # plt.xticks([])
    # plt.yticks([])
    ax.xaxis.set_major_formatter(plt.NullFormatter())
    ax.yaxis.set_major_formatter(plt.NullFormatter())
    plt.subplots_adjust(left=0.03, right=0.97, top=0.97, bottom=0.05)
    plt.xlim([0, dat.shape[1]])
    plt.ylim([0, dat.shape[0]])


# About the Colormaps
IMG_CMAP = plt.get_cmap('viridis')
IMG_CMAP.set_bad(color='black')
SEG_CMAP = random_cmap(ncolors=512, background_color=u'white')
SEG_CMAP.set_bad(color='white')
SEG_CMAP.set_under(color='white')

BLK = Greys_9.mpl_colormap
ORG = OrRd_9.mpl_colormap
BLU = Blues_9.mpl_colormap
GRN = YlGn_9.mpl_colormap
PUR = Purples_9.mpl_colormap

cmaplist = [
    '#000000', '#720026', '#A0213F', '#ce4257', '#E76154', '#ff9b54', '#ffd1b1'
]
cdict = {'red': [], 'green': [], 'blue': []}
cpoints = np.linspace(0, 1, len(cmaplist))
for i in range(len(cmaplist)):
    cdict['red'].append([
        cpoints[i],
        int(cmaplist[i][1:3], 16) / 256,
        int(cmaplist[i][1:3], 16) / 256
    ])
    cdict['green'].append([
        cpoints[i],
        int(cmaplist[i][3:5], 16) / 256,
        int(cmaplist[i][3:5], 16) / 256
    ])
    cdict['blue'].append([
        cpoints[i],
        int(cmaplist[i][5:7], 16) / 256,
        int(cmaplist[i][5:7], 16) / 256
    ])
autocmap = LinearSegmentedColormap('autocmap', cdict)
autocmap.set_under('k', alpha=0)


def display_image_colornorm(ax,
                            image,
                            cmap='Greys_r',
                            percentile=99.9,
                            vmin=None,
                            vmax=None,
                            scale_bar=True,
                            bar_length=10,
                            bar_fontsize=15,
                            pixel_scale=0.259,
                            box_alpha=1,
                            **kwargs):
    from astropy.visualization import PercentileInterval
    from matplotlib.colors import LogNorm, Normalize

    # Calculate the minimum and maximum values based on the central 99% of the data
    interval = PercentileInterval(percentile)
    if vmin is None:
        vmin = PercentileInterval(percentile).get_limits(image)[0]
    if vmax is None:
        vmax = PercentileInterval(percentile).get_limits(image)[1]

    ax.imshow(image,
              cmap=cmap,
              norm=LogNorm(vmin=vmin, vmax=vmax),
              aspect='auto',
              **kwargs)

    # Hide ticks and tick labels
    ax.tick_params(labelbottom=False,
                   labelleft=False,
                   axis=u'both',
                   which=u'both')  # length=0

    if scale_bar:
        scalebar = ScaleBar(pixel_scale,
                            "''",
                            dimension=ANGLE,
                            color='black',
                            box_alpha=box_alpha,
                            font_properties={'size': bar_fontsize},
                            location='lower left',
                            length_fraction=pixel_scale,
                            fixed_value=bar_length)
        ax.add_artist(scalebar)


def display_single(img,
                   pixel_scale=0.168,
                   physical_scale=None,
                   xsize=8,
                   ysize=8,
                   ax=None,
                   alpha=1.0,
                   stretch='arcsinh',
                   scale='zscale',
                   zmin=None,
                   zmax=None,
                   contrast=0.25,
                   no_negative=False,
                   lower_percentile=1.0,
                   upper_percentile=99.0,
                   cmap=IMG_CMAP,
                   scale_bar=True,
                   scale_bar_length=5.0,
                   scale_bar_fontsize=20,
                   scale_bar_y_offset=0.5,
                   scale_bar_color='w',
                   scale_bar_loc='left',
                   color_bar=False,
                   color_bar_loc=1,
                   color_bar_width='75%',
                   color_bar_height='5%',
                   color_bar_fontsize=18,
                   color_bar_color='w',
                   add_text=None,
                   text_fontsize=30,
                   text_color='w'):
    """Display single image.

    Parameters
    ----------
        img: np 2-D array for image

        xsize: int, default = 8
            Width of the image.

        ysize: int, default = 8
            Height of the image.

    """
    if ax is None:
        fig = plt.figure(figsize=(xsize, ysize))
        ax1 = fig.add_subplot(111)
    else:
        ax1 = ax
    ax1.grid(False)

    # Stretch option
    if img.ndim == 3:
        img_scale = img
        vmin, vmax = None, None
    else:
        if stretch.strip() == 'arcsinh':
            img_scale = np.arcsinh(img)
            if zmin is not None:
                zmin = np.arcsinh(zmin)
            if zmax is not None:
                zmax = np.arcsinh(zmax)
        elif stretch.strip() == 'log':
            if no_negative:
                img[img <= 0.0] = 1.0E-10
            img_scale = np.log(img)
            if zmin is not None:
                zmin = np.log(zmin)
            if zmax is not None:
                zmax = np.log(zmax)
        elif stretch.strip() == 'log10':
            if no_negative:
                img[img <= 0.0] = 1.0E-10
            img_scale = np.log10(img)
            if zmin is not None:
                zmin = np.log10(zmin)
            if zmax is not None:
                zmax = np.log10(zmax)
        elif stretch.strip() == 'linear':
            img_scale = img
        else:
            raise Exception("# Wrong stretch option.")

        # Scale option
        if scale.strip() == 'zscale':
            try:
                vmin, vmax = ZScaleInterval(
                    contrast=contrast).get_limits(img_scale)
            except IndexError:
                # TODO: Deal with problematic image
                vmin, vmax = -1.0, 1.0
        elif scale.strip() == 'percentile':
            try:
                vmin, vmax = AsymmetricPercentileInterval(
                    lower_percentile=lower_percentile,
                    upper_percentile=upper_percentile).get_limits(img_scale)
            except IndexError:
                # TODO: Deal with problematic image
                vmin, vmax = -1.0, 1.0
        elif scale.strip() == 'minmax':
            vmin, vmax = np.nanmin(img_scale), np.nanmax(img_scale)
        else:
            vmin, vmax = np.nanmin(img_scale), np.nanmax(img_scale)

        if zmin is not None:
            vmin = zmin
        if zmax is not None:
            vmax = zmax

    show = ax1.imshow(img_scale,
                      origin='lower',
                      cmap=cmap,
                      interpolation='none',
                      vmin=vmin,
                      vmax=vmax,
                      alpha=alpha,
                      aspect='auto')

    # Hide ticks and tick labels
    ax1.tick_params(labelbottom=False,
                    labelleft=False,
                    axis=u'both',
                    which=u'both')  # length=0

    # Put scale bar on the image
    if img.ndim == 3:
        img_size_x, img_size_y = img[:, :, 0].shape
    else:
        img_size_x, img_size_y = img.shape

    if physical_scale is not None:
        pixel_scale *= physical_scale

    if scale_bar:
        if scale_bar_loc == 'left':
            scale_bar_x_0 = int(img_size_x * 0.04)
            scale_bar_x_1 = int(img_size_x * 0.04 +
                                (scale_bar_length / pixel_scale))
        else:
            scale_bar_x_0 = int(img_size_x * 0.95 -
                                (scale_bar_length / pixel_scale))
            scale_bar_x_1 = int(img_size_x * 0.95)
        scale_bar_y = int(img_size_y * 0.10)
        scale_bar_text_x = (scale_bar_x_0 + scale_bar_x_1) / 2
        scale_bar_text_y = (scale_bar_y * scale_bar_y_offset)
        if physical_scale is not None:
            scale_bar_text = r'$%d\ \mathrm{kpc}$' % int(scale_bar_length)
        else:
            scale_bar_text = r'$%d^{\prime\prime}$' % int(scale_bar_length)
        scale_bar_text_size = scale_bar_fontsize

        ax1.plot([scale_bar_x_0, scale_bar_x_1], [scale_bar_y, scale_bar_y],
                 linewidth=3,
                 c=scale_bar_color,
                 alpha=1.0)
        ax1.text(scale_bar_text_x,
                 scale_bar_text_y,
                 scale_bar_text,
                 fontsize=scale_bar_text_size,
                 horizontalalignment='center',
                 color=scale_bar_color)
    if add_text is not None:
        text_x_0 = int(img_size_x * 0.08)
        text_y_0 = int(img_size_y * 0.80)
        ax1.text(text_x_0,
                 text_y_0,
                 r'$\mathrm{' + add_text + '}$',
                 fontsize=text_fontsize,
                 color=text_color)

    # Put a color bar on the image
    if color_bar:
        ax_cbar = inset_axes(ax1,
                             width=color_bar_width,
                             height=color_bar_height,
                             loc=color_bar_loc)
        if ax is None:
            cbar = plt.colorbar(show,
                                ax=ax1,
                                cax=ax_cbar,
                                orientation='horizontal')
        else:
            cbar = plt.colorbar(show,
                                ax=ax,
                                cax=ax_cbar,
                                orientation='horizontal')

        cbar.ax.xaxis.set_tick_params(color=color_bar_color)
        cbar.ax.yaxis.set_tick_params(color=color_bar_color)
        cbar.outline.set_edgecolor(color_bar_color)
        plt.setp(plt.getp(cbar.ax.axes, 'xticklabels'),
                 color=color_bar_color,
                 fontsize=color_bar_fontsize)
        plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'),
                 color=color_bar_color,
                 fontsize=color_bar_fontsize)

    if ax is None:
        return fig
    return ax1


def display_single2(img,
                    pixel_scale=0.168,
                    physical_scale=None,
                    xsize=8,
                    ysize=8,
                    ax=None,
                    alpha=1.0,
                    stretch='arcsinh',
                    scale='zscale',
                    zmin=None,
                    zmax=None,
                    contrast=0.25,
                    no_negative=False,
                    lower_percentile=1.0,
                    upper_percentile=99.0,
                    cmap=IMG_CMAP,
                    scale_bar=True,
                    bar_length=10,
                    bar_fontsize=12,
                    box_alpha=1,
                    color_bar=False,
                    color_bar_loc=1,
                    color_bar_width='75%',
                    color_bar_height='5%',
                    color_bar_fontsize=18,
                    color_bar_color='w',
                    add_text=None,
                    text_fontsize=15,
                    text_color='w'):
    """Display single image.

    Parameters
    ----------
        img: np 2-D array for image

        xsize: int, default = 8
            Width of the image.

        ysize: int, default = 8
            Height of the image.

    """
    if ax is None:
        fig = plt.figure(figsize=(xsize, ysize))
        ax1 = fig.add_subplot(111)
    else:
        ax1 = ax
    ax1.grid(False)

    # Stretch option
    if img.ndim == 3:
        img_scale = img
        vmin, vmax = None, None
    else:
        if stretch.strip() == 'arcsinh':
            img_scale = np.arcsinh(img)
            if zmin is not None:
                zmin = np.arcsinh(zmin)
            if zmax is not None:
                zmax = np.arcsinh(zmax)
        elif stretch.strip() == 'log':
            if no_negative:
                img[img <= 0.0] = 1.0E-10
            img_scale = np.log(img)
            if zmin is not None:
                zmin = np.log(zmin)
            if zmax is not None:
                zmax = np.log(zmax)
        elif stretch.strip() == 'log10':
            if no_negative:
                img[img <= 0.0] = 1.0E-10
            img_scale = np.log10(img)
            if zmin is not None:
                zmin = np.log10(zmin)
            if zmax is not None:
                zmax = np.log10(zmax)
        elif stretch.strip() == 'linear':
            img_scale = img
        else:
            raise Exception("# Wrong stretch option.")

        # Scale option
        if scale.strip() == 'zscale':
            try:
                vmin, vmax = ZScaleInterval(
                    contrast=contrast).get_limits(img_scale)
            except IndexError:
                # TODO: Deal with problematic image
                vmin, vmax = -1.0, 1.0
        elif scale.strip() == 'percentile':
            try:
                vmin, vmax = AsymmetricPercentileInterval(
                    lower_percentile=lower_percentile,
                    upper_percentile=upper_percentile).get_limits(img_scale)
            except IndexError:
                # TODO: Deal with problematic image
                vmin, vmax = -1.0, 1.0
        elif scale.strip() == 'minmax':
            vmin, vmax = np.nanmin(img_scale), np.nanmax(img_scale)
        else:
            vmin, vmax = np.nanmin(img_scale), np.nanmax(img_scale)

        if zmin is not None:
            vmin = zmin
        if zmax is not None:
            vmax = zmax

    show = ax1.imshow(img_scale,
                      origin='lower',
                      cmap=cmap,
                      interpolation='none',
                      vmin=vmin,
                      vmax=vmax,
                      alpha=alpha,
                      aspect='auto')

    # Hide ticks and tick labels
    ax1.tick_params(labelbottom=False,
                    labelleft=False,
                    axis=u'both',
                    which=u'both')  # length=0

    # Put scale bar on the image
    if img.ndim == 3:
        img_size_x, img_size_y = img[:, :, 0].shape
    else:
        img_size_x, img_size_y = img.shape

    if physical_scale is not None:
        pixel_scale *= physical_scale

    if scale_bar:
        scalebar = ScaleBar(pixel_scale,
                            "''",
                            dimension=ANGLE,
                            color='black',
                            box_alpha=box_alpha,
                            font_properties={'size': bar_fontsize},
                            location='lower left',
                            length_fraction=pixel_scale,
                            fixed_value=bar_length)
        ax1.add_artist(scalebar)

    if add_text is not None:
        text_x_0 = int(img_size_x * 0.08)
        text_y_0 = int(img_size_y * 0.80)
        ax1.text(text_x_0,
                 text_y_0,
                 r'$\mathrm{' + add_text + '}$',
                 fontsize=text_fontsize,
                 color=text_color)

    # Put a color bar on the image
    if color_bar:
        ax_cbar = inset_axes(ax1,
                             width=color_bar_width,
                             height=color_bar_height,
                             loc=color_bar_loc)
        if ax is None:
            cbar = plt.colorbar(show,
                                ax=ax1,
                                cax=ax_cbar,
                                orientation='horizontal')
        else:
            cbar = plt.colorbar(show,
                                ax=ax,
                                cax=ax_cbar,
                                orientation='horizontal')

        cbar.ax.xaxis.set_tick_params(color=color_bar_color)
        cbar.ax.yaxis.set_tick_params(color=color_bar_color)
        cbar.outline.set_edgecolor(color_bar_color)
        plt.setp(plt.getp(cbar.ax.axes, 'xticklabels'),
                 color=color_bar_color,
                 fontsize=color_bar_fontsize)
        plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'),
                 color=color_bar_color,
                 fontsize=color_bar_fontsize)

    if ax is None:
        return fig
    return ax1


def display_all(img_list,
                n_column=3,
                img_size=3.,
                hdu_index=None,
                label_list=None,
                cmap_list=None,
                label_x=0.1,
                label_y=0.9,
                fontsize=20,
                fontcolor='k',
                hdu_list=False,
                hdu_start=1,
                **kwargs):
    """Display a list of images."""
    if not isinstance(img_list, list):
        raise TypeError(
            "Provide a list of image to show or use display_single()")

    # Make a numpy array list if the input is HDUList
    if hdu_list:
        img_list = [
            img_list[ii].data for ii in np.arange(len(img_list))[hdu_start:]
        ]

    if cmap_list is not None:
        assert len(cmap_list) == len(img_list), "Wrong number of color maps!"

    if label_list is not None:
        assert len(label_list) == len(img_list), "Wrong number of labels!"

    # Number of image to show
    n_img = len(img_list)

    if n_img <= n_column:
        n_col = n_img
        n_row = 1
    else:
        n_col = n_column
        n_row = int(np.ceil(n_img / n_column))

    fig = plt.figure(figsize=(img_size * n_col, img_size * n_row))
    fig.subplots_adjust(left=0.,
                        right=1.,
                        bottom=0.,
                        top=1.,
                        wspace=0.,
                        hspace=0.)

    gs = gridspec.GridSpec(n_row, n_col)
    gs.update(wspace=0.0, hspace=0.00)

    for ii in range(n_img):
        if hdu_index is None:
            img_show = img_list[ii]
        else:
            img_show = img_list[ii][hdu_index].data

        ax = plt.subplot(gs[ii])
        if cmap_list is not None:
            ax = display_single(img_show, cmap=cmap_list[ii], ax=ax, **kwargs)
        else:
            ax = display_single(img_show, ax=ax, **kwargs)

        if label_list is not None:
            if len(label_list) != n_img:
                print("# Wrong number for labels!")
            else:
                ax.text(label_x,
                        label_y,
                        label_list[ii],
                        fontsize=fontsize,
                        transform=ax.transAxes,
                        color=fontcolor)

    return fig


def display_isophote(img, x0, y0, sma, ell, pa, ax, pixel_size=0.259):
    """Visualize the isophotes."""

    display_single(img,
                   ax=ax,
                   scale_bar=True,
                   pixel_scale=pixel_size,
                   cmap='Greys_r',
                   scale_bar_length=10,
                   scale='log10')

    for k in range(len(sma)):
        if k % 2 == 0:
            e = Ellipse(xy=(x0, y0),
                        height=sma[k] * 2.0,
                        width=sma[k] * 2.0 * (1.0 - ell[k]),
                        angle=pa[k])
            e.set_facecolor('none')
            e.set_edgecolor('#878ECD')
            e.set_alpha(1)
            e.set_linewidth(1.5)
            ax.add_artist(e)


def display_single_easy(ax,
                        image_data,
                        scale_bar=True,
                        bar_length=10,
                        box_alpha=1,
                        pixel_size=0.168):
    from astropy.visualization import simple_norm

    # Create an ImageNormalize object
    norm = simple_norm(image_data, 'log')

    # Display the image
    fig = plt.figure()
    im = ax.imshow(image_data, origin='lower', norm=norm, cmap='Greys_r')

    ax.xaxis.set_major_formatter(plt.NullFormatter())
    ax.yaxis.set_major_formatter(plt.NullFormatter())

    if scale_bar:
        scalebar = ScaleBar(pixel_size,
                            "''",
                            dimension=ANGLE,
                            color='black',
                            box_alpha=box_alpha,
                            font_properties={'size': 15},
                            location='lower left',
                            length_fraction=pixel_size,
                            fixed_value=bar_length)
        ax.add_artist(scalebar)


def display_isophote_LSB(ax,
                         img,
                         x0,
                         y0,
                         sma,
                         ell,
                         pa,
                         noise,
                         pixel_size=0.259,
                         scale_bar_length=50):
    """Visualize the isophotes using LSBImage."""

    LSBImage(ax=ax,
             dat=img,
             noise=noise,
             pixel_size=pixel_size,
             bar_length=scale_bar_length)

    for k in range(len(sma)):
        if (k % 2 == 0) & (k < 0.7 * len(sma)):
            e = Ellipse(xy=(x0, y0),
                        height=sma[k] * 2.0,
                        width=sma[k] * 2.0 * (1.0 - ell[k]),
                        angle=pa[k])
            e.set_facecolor('none')
            e.set_edgecolor('#878ECD')
            e.set_alpha(1)
            e.set_linewidth(1.5)
            ax.add_artist(e)

    for k in range(len(sma)):
        if k >= 0.7 * len(sma):
            e = Ellipse(xy=(x0, y0),
                        height=sma[k] * 2.0,
                        width=sma[k] * 2.0 * (1.0 - ell[k]),
                        angle=pa[k])
            e.set_facecolor('none')
            e.set_edgecolor('#878ECD')
            e.set_alpha(1)
            e.set_linewidth(1.5)
            ax.add_artist(e)


def easy_saveData_Tofits(data, header, savefile):
    hdu = fits.PrimaryHDU(data, header=header)

    hdul = fits.HDUList(hdu)
    hdul.writeto(savefile, overwrite=True)


def arcsec2kpc(x, D):
    """This function is for secondary axis to show the physical units, kpc.
       Note: when use this function, we should predefine a D, D is the distance of the objects.

    Args:
        x (acutually the theta): radius/arcsec

    Returns:
        [type]: [description]
    """
    Rkpc = (x * np.pi * D) / (18 * 36)

    return Rkpc


def kpc2arcsec(x, D):
    """The inverse function of the [arcsec2kpc] for secondary axis.

    Args:
        x ([type]): [description]

    Returns:
        [type]: [description]
    """
    Rsec = (x * 18 * 36) / (np.pi * D)

    return Rsec


def M2LToMass(BV, Mag_gal, Dist):
    # This function if for K band. the parameter is from KH13.
    logM2L = 1.055 * (BV) - 0.9402

    Mag_sun = 3.27

    logL_gal = (Mag_gal - Mag_sun) / (-2.5) - 2 * np.log10(
        1 / Dist) + 10  # the unit of L_gal is L_sun

    logM_gal = logM2L + logL_gal  # M_gal unit is M_sun

    return logM_gal


def M2LToMass_R(BR, Mag_gal, Dist):
    # This function if for K band. the parameter is from KH13.
    logM2L = 0.683 * (BR) - 0.523
    print('ML', logM2L)

    Mag_sun = 4.6

    logL_gal = (Mag_gal - Mag_sun) / (-2.5) - 2 * np.log10(
        1 / Dist) + 10  # the unit of L_gal is L_sun
    print('logL', logL_gal)

    logM_gal = logM2L + logL_gal  # M_gal unit is M_sun

    return logM_gal


def Ras2Rkpc(D, R_as):
    """ Transforme the arcsecond to kpc.

    Args:
        D (float): distance in Mpc
        R_as (float): Radius in arcsec.

    Returns:
        R_kpc: the radius in the unit of kpc.
    """

    return D * 1000 * R_as * np.pi / 180 / 60 / 60


def Rkpc2Ras(D, R_kpc):
    return R_kpc * 180 * 60 * 60 / np.pi / D / 1000


def KR(logre, a, b):
    """Show the kormendy relation. It is a linear function.

    Args:
        logre (float): log scale of effective radius.
        a (float): 
        b (float): 

    Returns:
        <mue>: the mean effective surface brightness of kormendy relation.
    """
    return a * logre + b


def fn(n):
    """The transformation factor from mue to mean mue.

    Args:
        n (float): Sersic index.

    Returns:
        fn: the transformation factor.
    """
    b = gammaincinv(2. * n, 0.5)

    fn = n * np.exp(b) / b**(2 * n) * gamma(2 * n)

    return fn


def mean_mue(mue, n):
    """ To calculate the mean mue for kormendy relation.

    Args:
        mue (float): effective surface brightness
        n (float): Seric index
    Returns:
        mean_mue_: mean effective surface brightness.
    """
    return mue - 2.5 * np.log10(fn(n))


def color2ML_profile(color, a, b):
    logM2L = a + b * color

    return logM2L


def mass_profile(cog_mag, logM2L, Dist, Mag_sun):
    """ To get the mass profiles based on stellar surface brightness profiles.

    Args:
        cog_mag (numpy array): The curve of growth in mag.
        color (numpy array): color profiles.
        a (float): mass-to-light ratio factor a.
        b (float): M/L facotr b.
        Dist (float): the distance of the galaxy. #!Notice the unit of Dist is Mpc.
        Mag_sun (float): the #! absolute magnitude of the Sun.

    Returns:
        mass: the stellar mass profiles.
    """

    logL_gal = (cog_mag - Mag_sun) / (-2.5) - 2 * np.log10(
        1 / Dist) + 10  # * the unit of L_gal is L_sun

    logM_gal = logM2L + logL_gal  # ! M_gal's unit is M_sun

    return logM_gal


def mass_density_profile(sma, logM_gal, Dist, ellipticity):
    """Get the mass surface density profiles.

    Args:
        sma (numpy array): semi-major axis radius in arcsec.
        logM_gal (numpy array): the stellar mass profiles.
        Dist (float): distance.
        ellipticity (float): the ellipticity of galaxy.
    """

    sma_arcsec = sma
    sma_kpc = Ras2Rkpc(Dist, sma_arcsec)

    mass_density_kpc = np.log10(10**logM_gal / (np.pi * sma_kpc**2 *
                                                (1 - ellipticity)))

    return mass_density_kpc


# Save 2-D numpy array to `fits`
def save_to_fits(img, fits_file, wcs=None, header=None, overwrite=True):
    """
    Save numpy 2-D arrays to `fits` file. (from `kungpao` https://github.com/dr-guangtou/kungpao)

    Parameters:
        img (numpy 2-D array): The 2-D array to be saved.
        fits_file (str): File name of `fits` file.
        wcs (``astropy.wcs.WCS`` object): World coordinate system (WCS) of this image.
        header (``astropy.io.fits.header`` or str): header of this image.
        overwrite (bool): Whether overwrite the file. Default is True.

    Returns:
        img_hdu (``astropy.fits.PrimaryHDU`` object)
    """
    img_hdu = fits.PrimaryHDU(img)

    if header is not None:
        img_hdu.header = header
        if wcs is not None:
            hdr = copy.deepcopy(header)
            wcs_header = wcs.to_header()
            import fnmatch
            for i in hdr:
                if i in wcs_header:
                    hdr[i] = wcs_header[i]
                if 'PC*' in wcs_header:
                    if fnmatch.fnmatch(i, 'CD?_?'):
                        hdr[i] = wcs_header['PC' + i.lstrip('CD')]
            img_hdu.header = hdr
    elif wcs is not None:
        wcs_header = wcs.to_header()
        wcs_header = fits.Header({'SIMPLE': True})
        wcs_header.update(NAXIS1=img.shape[1], NAXIS2=img.shape[0])
        for card in list(wcs.to_header().cards):
            wcs_header.append(card)
        img_hdu.header = wcs_header
    else:
        img_hdu = fits.PrimaryHDU(img)

    if os.path.islink(fits_file):
        os.unlink(fits_file)

    img_hdu.writeto(fits_file, overwrite=overwrite)
    return img_hdu


# Cutout image


def img_cutout(img,
               wcs,
               coord_1,
               coord_2,
               size=[60.0, 60.0],
               pixel_scale=0.168,
               pixel_unit=False,
               img_header=None,
               prefix='img_cutout',
               out_dir=None,
               save=True):
    """
    Generate image cutout with updated WCS information. (From ``kungpao`` https://github.com/dr-guangtou/kungpao) 

    Parameters:
        img (numpy 2-D array): image array.
        wcs (``astropy.wcs.WCS`` object): WCS of input image array.
        coord_1 (float): ``ra`` or ``x`` of the cutout center.
        coord_2 (float): ``dec`` or ``y`` of the cutout center.
        size (array): image size, such as (800, 1000), in arcsec unit by default.
        pixel_scale (float): pixel size, in the unit of "arcsec/pixel".
        pixel_unit (bool):  When True, ``coord_1``, ``coord_2`` becomes ``X``, ``Y`` pixel coordinates. 
            ``size`` will also be treated as in pixels.
        img_header: The header of input image, typically ``astropy.io.fits.header`` object.
            Provide the haeder in case you can save the infomation in this header to the new header.
        prefix (str): Prefix of output files.
        out_dir (str): Directory of output files. Default is the current folder.
        save (bool): Whether save the cutout image.

    Returns: 
        :
            cutout (numpy 2-D array): the cutout image.

            [cen_pos, dx, dy]: a list contains center position and ``dx``, ``dy``.

            cutout_header: Header of cutout image.
    """

    from astropy.nddata import Cutout2D
    if not pixel_unit:
        # img_size in unit of arcsec
        cutout_size = np.asarray(size) / pixel_scale
        cen_x, cen_y = wcs.all_world2pix(coord_1, coord_2, 0)
    else:
        cutout_size = np.asarray(size)
        cen_x, cen_y = coord_1, coord_2

    cen_pos = (int(cen_x), int(cen_y))
    dx = -1.0 * (cen_x - int(cen_x))
    dy = -1.0 * (cen_y - int(cen_y))

    # Generate cutout
    cutout = Cutout2D(img,
                      cen_pos,
                      cutout_size,
                      wcs=wcs,
                      mode='partial',
                      fill_value=0)

    # Update the header
    cutout_header = cutout.wcs.to_header()
    if img_header is not None:
        if 'COMMENT' in img_header:
            del img_header['COMMENT']
        intersect = [k for k in img_header if k not in cutout_header]
        for keyword in intersect:
            cutout_header.set(keyword, img_header[keyword],
                              img_header.comments[keyword])

    if 'PC1_1' in dict(cutout_header).keys():
        cutout_header['CD1_1'] = cutout_header['PC1_1']
        # cutout_header['CD1_2'] = cutout_header['PC1_2']
        # cutout_header['CD2_1'] = cutout_header['PC2_1']
        cutout_header['CD2_2'] = cutout_header['PC2_2']
        cutout_header['CDELT1'] = cutout_header['CD1_1']
        cutout_header['CDELT2'] = cutout_header['CD2_2']
        cutout_header.pop('PC1_1')
        # cutout_header.pop('PC2_1')
        # cutout_header.pop('PC1_2')
        cutout_header.pop('PC2_2')
        # cutout_header.pop('CDELT1')
        # cutout_header.pop('CDELT2')

    # Build a HDU
    hdu = fits.PrimaryHDU(header=cutout_header)
    hdu.data = cutout.data
    # hdu.data = np.flipud(cutout.data)
    # Save FITS image
    if save:
        fits_file = prefix + '.fits'
        if out_dir is not None:
            fits_file = os.path.join(out_dir, fits_file)

        hdu.writeto(fits_file, overwrite=True)

    return cutout, [cen_pos, dx, dy], cutout_header


def extract_obj(img,
                mask=None,
                b=64,
                f=3,
                sigma=5,
                pixel_scale=0.168,
                minarea=5,
                convolve=False,
                convolve_kernel='tophat',
                conv_radius=3,
                deblend_nthresh=32,
                deblend_cont=0.005,
                clean_param=1.0,
                sky_subtract=False,
                flux_auto=True,
                flux_aper=None,
                show_fig=False,
                verbose=True,
                logger=None,
                **kwargs):
    '''
    Extract objects for a given image using ``sep`` (a Python-wrapped ``SExtractor``). 
    For more details, please check http://sep.readthedocs.io and documentation of SExtractor.

    Parameters:
        img (numpy 2-D array): input image
        mask (numpy 2-D array): image mask
        b (float): size of box
        f (float): size of convolving kernel
        sigma (float): detection threshold
        pixel_scale (float): default is 0.168 (HSC pixel size). This only affect the figure scale bar.
        minarea (float): minimum number of connected pixels
        deblend_nthresh (float): Number of thresholds used for object deblending
        deblend_cont (float): Minimum contrast ratio used for object deblending. Set to 1.0 to disable deblending. 
        clean_param (float): Cleaning parameter (see SExtractor manual)
        sky_subtract (bool): whether subtract sky before extract objects (this will affect the measured flux).
        flux_auto (bool): whether return AUTO photometry (see SExtractor manual)
        flux_aper (list): such as [3, 6], which gives flux within [3 pix, 6 pix] annulus.

    Returns:
        :
            objects: `astropy` Table, containing the positions,
                shapes and other properties of extracted objects.

            segmap: 2-D numpy array, segmentation map
    '''
    import sep

    sep.set_extract_pixstack(1e6)
    # Subtract a mean sky value to achieve better object detection
    b = b  # Box size
    f = f  # Filter width

    # if convolve:
    #     from astropy.convolution import convolve, Gaussian2DKernel
    #     img = convolve(img.astype(float), Gaussian2DKernel(conv_radius))

    try:
        bkg = sep.Background(img, bw=b, bh=b, fw=f, fh=f)
    except ValueError as e:
        img = img.byteswap().newbyteorder()
        bkg = sep.Background(img, bw=b, bh=b, fw=f, fh=f)

    data_sub = img - bkg.back()

    sigma = sigma

    if sky_subtract:
        input_data = data_sub
    else:
        input_data = img

    if convolve:
        if convolve_kernel == 'gaussian':
            from astropy.convolution import Gaussian2DKernel, convolve
            input_data = convolve(input_data.astype(float),
                                  Gaussian2DKernel(conv_radius))
            bkg = sep.Background(input_data, bw=b, bh=b, fw=f, fh=f)
            input_data -= bkg.globalback

        elif convolve_kernel == 'tophat':
            from astropy.convolution import Tophat2DKernel, convolve
            input_data = convolve(input_data.astype(float),
                                  Tophat2DKernel(conv_radius))
            bkg = sep.Background(input_data, bw=b, bh=b, fw=f, fh=f)
            input_data -= bkg.globalback

    objects, segmap = sep.extract(input_data,
                                  sigma,
                                  mask=mask,
                                  err=bkg.globalrms,
                                  segmentation_map=True,
                                  filter_type='matched',
                                  deblend_nthresh=deblend_nthresh,
                                  deblend_cont=deblend_cont,
                                  clean=True,
                                  clean_param=clean_param,
                                  minarea=minarea,
                                  **kwargs)

    if verbose:
        if logger is not None:
            logger.info("    Detected %d objects" % len(objects))
        print("    Detected %d objects" % len(objects))
    objects = Table(objects)
    objects.add_column(Column(data=np.arange(len(objects)), name='index'))
    # Maximum flux, defined as flux within 6 * `a` (semi-major axis) in radius.
    objects.add_column(
        Column(data=sep.sum_circle(input_data, objects['x'], objects['y'],
                                   6. * objects['a'])[0],
               name='flux_max'))
    # Add FWHM estimated from 'a' and 'b'.
    # This is suggested here: https://github.com/kbarbary/sep/issues/34
    objects.add_column(
        Column(data=2 *
               np.sqrt(np.log(2) * (objects['a']**2 + objects['b']**2)),
               name='fwhm_custom'))

    # Measure R30, R50, R80
    temp = sep.flux_radius(input_data, objects['x'], objects['y'],
                           6. * objects['a'], [0.2, 0.3, 0.5, 0.8, 0.9])[0]
    objects.add_column(Column(data=temp[:, 0], name='R20'))
    objects.add_column(Column(data=temp[:, 0], name='R30'))
    objects.add_column(Column(data=temp[:, 1], name='R50'))
    objects.add_column(Column(data=temp[:, 2], name='R80'))
    objects.add_column(Column(data=temp[:, 3], name='R90'))

    # Use Kron radius to calculate FLUX_AUTO in SourceExtractor.
    # Here PHOT_PARAMETER = 2.5, 3.5
    if flux_auto:
        kronrad, krflag = sep.kron_radius(input_data, objects['x'],
                                          objects['y'], objects['a'],
                                          objects['b'], objects['theta'], 6.0)
        flux, fluxerr, flag = sep.sum_circle(input_data,
                                             objects['x'],
                                             objects['y'],
                                             2.5 * (kronrad),
                                             subpix=1)
        flag |= krflag  # combine flags into 'flag'

        r_min = 1.75  # minimum diameter = 3.5
        use_circle = kronrad * np.sqrt(objects['a'] * objects['b']) < r_min
        cflux, cfluxerr, cflag = sep.sum_circle(input_data,
                                                objects['x'][use_circle],
                                                objects['y'][use_circle],
                                                r_min,
                                                subpix=1)
        flux[use_circle] = cflux
        fluxerr[use_circle] = cfluxerr
        flag[use_circle] = cflag
        objects.add_column(Column(data=flux, name='flux_auto'))
        objects.add_column(Column(data=kronrad, name='kron_rad'))

    if flux_aper is not None:
        if len(flux_aper) != 2:
            raise ValueError('"flux_aper" must be a list with length = 2.')
        objects.add_column(
            Column(data=sep.sum_circle(input_data, objects['x'], objects['y'],
                                       flux_aper[0])[0],
                   name='flux_aper_1'))
        objects.add_column(
            Column(data=sep.sum_circle(input_data, objects['x'], objects['y'],
                                       flux_aper[1])[0],
                   name='flux_aper_2'))
        objects.add_column(
            Column(data=sep.sum_circann(input_data, objects['x'], objects['y'],
                                        flux_aper[0], flux_aper[1])[0],
                   name='flux_ann'))

    # plot background-subtracted image
    if show_fig:
        fig, ax = plt.subplots(1, 2, figsize=(12, 6))
        if min(input_data.shape) * pixel_scale < 5:
            scale_bar_length = 0.5
        if min(input_data.shape) * pixel_scale < 30:
            scale_bar_length = 1
        elif min(input_data.shape) * pixel_scale > 100:
            scale_bar_length = 61
        else:
            scale_bar_length = 10
        ax[0] = display_single(input_data,
                               ax=ax[0],
                               scale_bar_length=scale_bar_length,
                               pixel_scale=pixel_scale)
        if mask is not None:
            ax[0].imshow(mask.astype(float),
                         origin='lower',
                         alpha=0.1,
                         cmap='Greys_r')
        from matplotlib.patches import Ellipse

        # plot an ellipse for each object
        for obj in objects:
            e = Ellipse(xy=(obj['x'], obj['y']),
                        width=5 * obj['a'],
                        height=5 * obj['b'],
                        angle=obj['theta'] * 180. / np.pi)
            e.set_facecolor('none')
            e.set_edgecolor('red')
            ax[0].add_artist(e)
        ax[1] = display_single(segmap,
                               scale='linear',
                               cmap=SEG_CMAP,
                               ax=ax[1],
                               scale_bar_length=scale_bar_length)
        # plt.savefig('./extract_obj.png', bbox_inches='tight')
        return objects, segmap, fig
    return objects, segmap


def seg_remove_cen_obj(seg):
    """Remove the central object from the segmentation."""
    seg_copy = copy.deepcopy(seg)
    seg_copy[seg == seg[int(seg.shape[1] / 2.0), int(seg.shape[0] / 2.0)]] = 0

    return seg_copy


def seg_remove_obj(seg, x, y):
    """Remove an object from the segmentation given its coordinate.

    Parameters
    ----------
    seg     : 2-D data array, segmentation mask
    x, y    : int, coordinates

    TODO:
        Should be absorbed by objects for segmentation image
    """
    seg_copy = copy.deepcopy(seg)
    seg_copy[seg == seg[int(y), int(x)]] = 0

    return seg_copy


def seg_index_obj(seg, x, y):
    """Remove the index array for an object given its location.

    TODO
    ----
        Should be absorbed by objects for segmentation image

    """
    #TODO should check the order of int(x) and int(y). Originally int(x) and int(y) in kungpao.
    obj = seg[int(y), int(x)]
    if obj == 0:
        return None

    return seg == obj


def increase_mask_regions(mask, method='uniform', size=7, mask_threshold=0.01):
    """Increase the size of the mask regions using smoothing algorithm. From jiaxuanli"""
    mask_arr = mask.astype('int16')
    mask_arr[mask_arr > 0] = 100

    if method == 'uniform' or method == 'box':
        mask_new = ndimage.uniform_filter(mask_arr, size=size)
    elif method == 'gaussian':
        mask_new = ndimage.gaussian_filter(mask_arr, sigma=size, order=0)
    else:
        raise ValueError("Wrong method. Should be uniform or gaussian.")

    mask_new[mask_new < mask_threshold] = 0
    mask_new[mask_new >= mask_threshold] = 1

    return mask_new.astype('uint8')


def seg_to_mask(seg, sigma=5.0, msk_max=1000.0, msk_thr=0.01):
    """Convert the segmentation array into an array. 
       Basically it is a Gaussian smoothing of the segmentation array. From kungpao.

    Parameters
    ----------
        sigma:  Sigma of the Gaussian Kernel

    """
    from scipy import ndimage

    # Convolve the mask image with a gaussian kernel
    msk_conv = ndimage.gaussian_filter(((seg.copy() > 0) * msk_max),
                                       sigma=sigma,
                                       order=0)
    msk_bool = msk_conv > (msk_thr * msk_max)

    return msk_bool.astype('uint8')


def _image_gaia_stars_tigress(image,
                              wcs,
                              pixel_scale=0.168,
                              mask_a=694.7,
                              mask_b=3.5,
                              verbose=True,
                              visual=False,
                              size_buffer=1.4,
                              logger=None):
    """
    Search for bright stars using GAIA catalogs on Tigress (`/tigress/HSC/refcats/htm/gaia_dr2_20200414`).
    For more information, see https://community.lsst.org/t/gaia-dr2-reference-catalog-in-lsst-format/3901.
    This function requires `lsstpipe`.

    Parameters:
        image (numpy 2-D array): input image.
        wcs (`astropy.wcs` object): WCS of the input image.
        pixel_scale (float): default is 0.168 (HSC pixel size). This only affect the figure scale bar.
        mask_a (float): a scaling factor for the size of the plotted star, larger value means larger circle will be plotted.
        mask_b (float): a scale size for the plotted star, larger value gives larger circle. 
        visual (bool): whether display the matched Gaia stars.

    Return: 
        gaia_results (`astropy.table.Table` object): a catalog of matched stars.
    """
    # Central coordinate
    ra_cen, dec_cen = wcs.all_pix2world(image.shape[1] / 2, image.shape[0] / 2,
                                        0)
    img_cen_ra_dec = SkyCoord(ra_cen,
                              dec_cen,
                              unit=('deg', 'deg'),
                              frame='icrs')

    # Width and height of the search box
    img_ra_size = Quantity(pixel_scale * (image.shape)[1] * size_buffer,
                           u.arcsec).to(u.degree)
    img_dec_size = Quantity(pixel_scale * (image.shape)[0] * size_buffer,
                            u.arcsec).to(u.degree)

    # Search for stars in Gaia catatlogs, which are stored in
    # `/tigress/HSC/refcats/htm/gaia_dr2_20200414`.
    try:
        import lsst.geom as geom
        from lsst.meas.algorithms.htmIndexer import HtmIndexer

        def getShards(ra, dec, radius):
            htm = HtmIndexer(depth=7)

            afw_coords = geom.SpherePoint(geom.Angle(ra, geom.degrees),
                                          geom.Angle(dec, geom.degrees))

            shards, onBoundary = htm.getShardIds(afw_coords,
                                                 radius * geom.degrees)
            return shards

    except ImportError as e:
        # Output expected ImportErrors.
        if logger is not None:
            logger.error(e)
            logger.error(
                'LSST Pipe must be installed to query Gaia stars on Tigress.')
        print(e)
        print('LSST Pipe must be installed to query Gaia stars on Tigress.')

    # find out the Shard ID of target area in the HTM (Hierarchical triangular mesh) system
    if logger is not None:
        logger.info('    Taking Gaia catalogs stored in `Tigress`')
    print('    Taking Gaia catalogs stored in `Tigress`')

    shards = getShards(ra_cen, dec_cen,
                       max(img_ra_size, img_dec_size).to(u.degree).value)
    cat = vstack([
        Table.read(f'/tigress/HSC/refcats/htm/gaia_dr2_20200414/{index}.fits')
        for index in shards
    ])
    cat['coord_ra'] = cat['coord_ra'].to(u.degree)
    # why GAIA coordinates are in RADIAN???
    cat['coord_dec'] = cat['coord_dec'].to(u.degree)

    # Trim this catalog a little bit
    # Ref: https://github.com/MerianSurvey/caterpillar/blob/main/caterpillar/catalog.py
    if cat:  # if not empty
        gaia_results = cat[
            (cat['coord_ra'] > img_cen_ra_dec.ra - img_ra_size / 2)
            & (cat['coord_ra'] < img_cen_ra_dec.ra + img_ra_size / 2) &
            (cat['coord_dec'] > img_cen_ra_dec.dec - img_dec_size / 2) &
            (cat['coord_dec'] < img_cen_ra_dec.dec + img_dec_size / 2)]
        gaia_results.rename_columns(['coord_ra', 'coord_dec'], ['ra', 'dec'])

        gaia_results['phot_g_mean_mag'] = -2.5 * \
                                          np.log10(
                                              (gaia_results['phot_g_mean_flux'] / (3631 * u.Jy)))  # AB magnitude

        # Convert the (RA, Dec) of stars into pixel coordinate using WCS
        x_gaia, y_gaia = wcs.all_world2pix(gaia_results['ra'],
                                           gaia_results['dec'], 0)

        # Generate mask for each star
        rmask_gaia_arcsec = mask_a * np.exp(
            -gaia_results['phot_g_mean_mag'] / mask_b)

        # Update the catalog
        gaia_results.add_column(Column(data=x_gaia, name='x_pix'))
        gaia_results.add_column(Column(data=y_gaia, name='y_pix'))
        gaia_results.add_column(
            Column(data=rmask_gaia_arcsec, name='rmask_arcsec'))

        if visual:
            fig = plt.figure(figsize=(8, 8))
            ax1 = fig.add_subplot(111)

            ax1 = display_single(image, ax=ax1)
            # Plot an ellipse for each object
            for star in gaia_results:
                smask = mplEllip(
                    xy=(star['x_pix'], star['y_pix']),
                    width=(2.0 * star['rmask_arcsec'] / pixel_scale),
                    height=(2.0 * star['rmask_arcsec'] / pixel_scale),
                    angle=0.0)
                smask.set_facecolor(ORG(0.2))
                smask.set_edgecolor(ORG(1.0))
                smask.set_alpha(0.3)
                ax1.add_artist(smask)

            # Show stars
            ax1.scatter(gaia_results['x_pix'],
                        gaia_results['y_pix'],
                        color=ORG(1.0),
                        s=100,
                        alpha=0.9,
                        marker='+')

            ax1.set_xlim(0, image.shape[1])
            ax1.set_ylim(0, image.shape[0])

        return gaia_results

    return None


def image_gaia_stars(image,
                     wcs,
                     pixel_scale=0.168,
                     mask_a=694.7,
                     mask_b=3.5,
                     verbose=False,
                     visual=False,
                     size_buffer=1.4,
                     tap_url=None):
    """
    Search for bright stars using GAIA catalog. From https://github.com/dr-guangtou/kungpao.

    Parameters:
        image (numpy 2-D array): input image.
        wcs (`astropy.wcs` object): WCS of the input image.
        pixel_scale (float): default is 0.168 (HSC pixel size). This only affect the figure scale bar.
        mask_a (float): a scaling factor for the size of the plotted star, larger value means larger circle will be plotted.
        mask_b (float): a scale size for the plotted star, larger value gives larger circle. 
        visual (bool): whether display the matched Gaia stars.

    Return: 
        gaia_results (`astropy.table.Table` object): a catalog of matched stars.
    """
    # Central coordinate
    ra_cen, dec_cen = wcs.all_pix2world(image.shape[0] / 2, image.shape[1] / 2,
                                        0)
    img_cen_ra_dec = SkyCoord(ra_cen,
                              dec_cen,
                              unit=('deg', 'deg'),
                              frame='icrs')

    # Width and height of the search box
    img_search_x = Quantity(pixel_scale * (image.shape)[0] * size_buffer,
                            u.arcsec)
    img_search_y = Quantity(pixel_scale * (image.shape)[1] * size_buffer,
                            u.arcsec)

    # Search for stars
    if tap_url is not None:
        with suppress_stdout():
            from astroquery.gaia import GaiaClass, TapPlus
            Gaia = GaiaClass(TapPlus(url=tap_url))

            gaia_results = Gaia.query_object_async(coordinate=img_cen_ra_dec,
                                                   width=img_search_x,
                                                   height=img_search_y,
                                                   verbose=verbose)
    else:
        with suppress_stdout():
            from astroquery.gaia import Gaia

            gaia_results = Gaia.query_object_async(coordinate=img_cen_ra_dec,
                                                   width=img_search_x,
                                                   height=img_search_y,
                                                   verbose=verbose)

    if gaia_results:
        # Convert the (RA, Dec) of stars into pixel coordinate
        ra_gaia = np.asarray(gaia_results['ra'])
        dec_gaia = np.asarray(gaia_results['dec'])
        x_gaia, y_gaia = wcs.all_world2pix(ra_gaia, dec_gaia, 0)

        # Generate mask for each star
        rmask_gaia_arcsec = mask_a * np.exp(
            -gaia_results['phot_g_mean_mag'] / mask_b)

        # Update the catalog
        gaia_results.add_column(Column(data=x_gaia, name='x_pix'))
        gaia_results.add_column(Column(data=y_gaia, name='y_pix'))
        gaia_results.add_column(
            Column(data=rmask_gaia_arcsec, name='rmask_arcsec'))

        if visual:
            fig = plt.figure(figsize=(8, 8))
            ax1 = fig.add_subplot(111)

            ax1 = display_single(image, ax=ax1)
            # Plot an ellipse for each object
            for star in gaia_results:
                smask = mplEllip(
                    xy=(star['x_pix'], star['y_pix']),
                    width=(2.0 * star['rmask_arcsec'] / pixel_scale),
                    height=(2.0 * star['rmask_arcsec'] / pixel_scale),
                    angle=0.0)
                smask.set_facecolor(ORG(0.2))
                smask.set_edgecolor(ORG(1.0))
                smask.set_alpha(0.3)
                ax1.add_artist(smask)

            # Show stars
            ax1.scatter(gaia_results['x_pix'],
                        gaia_results['y_pix'],
                        color=ORG(1.0),
                        s=100,
                        alpha=0.9,
                        marker='+')

            ax1.set_xlim(0, image.shape[0])
            ax1.set_ylim(0, image.shape[1])

        return gaia_results

    return None


def gaia_star_mask(img,
                   wcs,
                   gaia_stars=None,
                   pixel_scale=0.168,
                   mask_a=694.7,
                   mask_b=3.5,
                   size_buffer=1.4,
                   gaia_bright=18.0,
                   factor_b=1.3,
                   factor_f=1.9,
                   tigress=False):
    """Find stars using Gaia and mask them out if necessary. From https://github.com/dr-guangtou/kungpao.

    Using the stars found in the GAIA TAP catalog, we build a bright star mask following
    similar procedure in Coupon et al. (2017).

    We separate the GAIA stars into bright (G <= 18.0) and faint (G > 18.0) groups, and
    apply different parameters to build the mask.

    Parameters:
        img (numpy 2-D array): input image.
        wcs (`astropy.wcs` object): WCS of the input image.
        pixel_scale (float): default is 0.168 (HSC pixel size). This only affect the figure scale bar.
        mask_a (float): a scale factor for the size of the plotted star, larger value means larger circle will be plotted.
        mask_b (float): a scale size for the plotted star, larger value gives larger circle. 
        gaia_bright (float): a threshold above which are classified as bright stars.
        factor_b (float): a scale size of mask for bright stars. Larger value gives smaller mask size.
        factor_f (float): a scale size of mask for faint stars. Larger value gives smaller mask size.
        tigress (bool): whether take Gaia catalogs on Tigress

    Return: 
        msk_star (numpy 2-D array): the masked pixels are marked by one.

    """
    if gaia_stars is None:
        if tigress:
            gaia_stars = _image_gaia_stars_tigress(img,
                                                   wcs,
                                                   pixel_scale=pixel_scale,
                                                   mask_a=mask_a,
                                                   mask_b=mask_b,
                                                   verbose=False,
                                                   visual=False,
                                                   size_buffer=size_buffer)
        else:
            gaia_stars = image_gaia_stars(img,
                                          wcs,
                                          pixel_scale=pixel_scale,
                                          mask_a=mask_a,
                                          mask_b=mask_b,
                                          verbose=False,
                                          visual=False,
                                          size_buffer=size_buffer)
        if gaia_stars is not None:
            print(f'    {len(gaia_stars)} stars from Gaia are masked!')
        else:  # does not find Gaia stars
            print('    No Gaia stars are masked.')
    else:
        print(f'    {len(gaia_stars)} stars from Gaia are masked!')

    # Make a mask image
    msk_star = np.zeros(img.shape).astype('uint8')

    if gaia_stars is not None:
        gaia_b = gaia_stars[gaia_stars['phot_g_mean_mag'] <= gaia_bright]
        sep.mask_ellipse(msk_star,
                         gaia_b['x_pix'],
                         gaia_b['y_pix'],
                         gaia_b['rmask_arcsec'] / factor_b / pixel_scale,
                         gaia_b['rmask_arcsec'] / factor_b / pixel_scale,
                         0.0,
                         r=1.0)

        gaia_f = gaia_stars[gaia_stars['phot_g_mean_mag'] > gaia_bright]
        sep.mask_ellipse(msk_star,
                         gaia_f['x_pix'],
                         gaia_f['y_pix'],
                         gaia_f['rmask_arcsec'] / factor_f / pixel_scale,
                         gaia_f['rmask_arcsec'] / factor_f / pixel_scale,
                         0.0,
                         r=1.0)

        return gaia_stars, msk_star

    return None, msk_star


def create_circular_mask(img, center=None, radius=None):
    """Create a circular mask to apply to an image.
    
    Based on https://stackoverflow.com/questions/44865023/how-can-i-create-a-circular-mask-for-a-numpy-array
    """
    h, w = img.shape

    if center is None:  # use the middle of the image
        center = (int(w / 2), int(h / 2))
    if radius is None:  # use the smallest distance between the center and image walls
        radius = min(center[0], center[1], w - center[0], h - center[1])

    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y - center[1])**2)

    mask = dist_from_center <= radius
    return mask


def create_elliptical_mask(image_data,
                           center=None,
                           radius=None,
                           axis_ratio=1,
                           pa=0):
    from astropy.coordinates import Angle
    from regions import EllipsePixelRegion, PixCoord

    img_copy = copy.deepcopy(image_data)

    # ellipse mask
    reg = EllipsePixelRegion(PixCoord(img_copy.shape[0] / 2,
                                      img_copy.shape[1] / 2),
                             width=radius * 2,
                             height=axis_ratio * radius * 2,
                             angle=Angle(pa, 'deg'))

    ellipseGal_mask_ori = reg.to_mask()
    ellipseGal_mask_img = ellipseGal_mask_ori.to_image(image_data.shape)

    cen_mask = ellipseGal_mask_img == 1
    return cen_mask

def bulge_mask(image_data, redshift, r_phy = 1, pixel_scale=0.168, cosmo='Planck18'):
    from Taiji.imtools import create_circular_mask
    from Taiji.utils import r_phy_to_ang
    
    r_ang = r_phy_to_ang(redshift=redshift, r_phy=r_phy, cosmo=cosmo).value/pixel_scale
    
    bulge_mask = create_circular_mask(image_data, radius=r_ang)
    
    return bulge_mask

def remove_consecutive(sma_ap, index_original, con_number=2):
    """Remove continuous unnecesary apertures in SBPs.

    Args:
        sma_ap (_type_): _description_
        index_original (_type_): _description_
        con_number (int, optional): _description_. Defaults to 2.

    Returns:
        _type_: _description_
    """
    index_left = np.argwhere(index_original)
    index_left_array = np.array(
        [index_left[i][0] for i in range(len(index_left))])
    diff_index = np.diff(index_left_array)

    if len(sma_ap[np.argwhere(diff_index > (con_number + 1))]) > 0:
        sma_stop_lowsn = (np.array(sma_ap)[np.argwhere(
            diff_index > con_number + 1)])[0][0]
        print(sma_stop_lowsn)
        new_index = np.logical_and(index_original, sma_ap <= sma_stop_lowsn)
    else:
        sma_stop_lowsn = 'well'
        new_index = index_original
    return (new_index, sma_stop_lowsn)


def padding_PSF(psf_list):
    '''
    If the sizes of HSC PSF in all bands are not the same, this function pads the smaller PSFs.

    Parameters:
        psf_list: a list returned by `unagi.task.hsc_psf` function

    Returns:
        psf_pad: a list including padded PSFs. They now share the same size.
    '''
    # Padding PSF cutouts from HSC
    max_len = max([max(psf[0].data.shape) for psf in psf_list])
    psf_pad = []
    for psf in psf_list:
        y_len, x_len = psf[0].data.shape
        dy = (max_len - y_len) // 2
        dx = (max_len - x_len) // 2
        temp = np.pad(psf[0].data.astype('float'), ((dy, dy), (dx, dx)),
                      'constant',
                      constant_values=0)
        if temp.shape == (max_len, max_len):
            psf_pad.append(temp)
        else:
            raise ValueError('Wrong size!')


def deltaf_binomial(f, N):
    """The error bar of the fraction of different types.

    Args:
        f (float or array): the fraction.
        N (float or array): the number in each bin.

    Returns:
        error bar: the error bar of the fraction.
    """
    return np.sqrt(f * (1 - f) / N)


def Running_median(X, Y):
    """To derive the runing median.

    Args:
        X (array): _description_
        Y (array): _description_
        total_bins (int, optional): _description_. Defaults to 10.
    TODO: Using astropy.stats.scott_bin_width to determine the number of bins. And use scipy.stats.binned_statistic to calculate the median (https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.binned_statistic.html).
    """
    from astropy.stats import scott_bin_width

    hist_bins_scott = scott_bin_width(X[~np.isnan(X)], return_bins=True)

    total_bins = len(hist_bins_scott[1]) - 1
    print(total_bins)

    bins = np.linspace(np.nanmin(X), np.nanmax(X), total_bins)
    delta = bins[1] - bins[0]
    idx = np.digitize(X, bins)
    running_median = [np.nanmedian(Y[idx == k]) for k in range(total_bins)]
    running_std = [np.std(Y[idx == k]) for k in range(total_bins)]

    new_x = bins - delta / 2
    new_y = running_median

    return np.array(new_x), np.array(new_y), np.array(running_std)

def extract_fix_isophotes(image=None, xcen=None, ycen=None, initsma=None, eps=None, pa=None, step=None, 
                          linear_growth=False, minsma=None, maxsma=None, silent=False, integrmode='bilinear', sclip=3.0, nclip=3, mask=None):
    """
    Function to extract surface brightness profile with fixed center, ellipticity, and position angle.
    
    Note: init_sma should be large, e.g., init_sma = 2r50 !!! Otherwise, the result will be wrong, i.e., init_sma=1 the instensity profile will be wrong. See https://github.com/astropy/photutils/issues/1222.
    """
    
    from photutils.isophote import (EllipseGeometry, EllipseSample, Isophote,
                                    IsophoteList)
    
    syntax = "syntax: results = extract_fix_isophotes(image=, xcen=, ycen=, initsma=, eps=, pa=, step=, linear_growth=False/True, minsma=None, maxsma=None, silent=False/True, integrmode='bilinear', sclip=3.0, nclip=3; minsma maxsma are optional)"
    if (None in [xcen, ycen, initsma, eps, pa, step]) or (image is None):
        print(syntax)
        return []
    print(syntax) if silent == False else print("")
    
    if mask is not None:
        image_data = copy.deepcopy(image)
        image_data_masked = np.ma.masked_where(mask == 1, image_data)
        image = image_data_masked
    
    minsma = minsma if minsma is not None else 0.5
    maxsma = maxsma if maxsma is not None else max(np.shape(image))/2*1.3
    isophote_list = []

    geometry = EllipseGeometry(xcen, ycen, initsma, eps, pa, astep=step, linear_growth=linear_growth, 
                               fix_center=True, fix_pa=True, fix_eps=True)
    
    sma = initsma
    while True:
        sample = EllipseSample(image, sma, geometry=geometry, integrmode=integrmode, sclip=sclip, nclip=nclip)

        sample.update(geometry.fix)
        isophote = Isophote(sample, 0, True, stop_code=4)
        isophote_list.append(isophote)
        sma = isophote.sample.geometry.update_sma(step)
        if maxsma and sma >= maxsma:
            break

    first_isophote = isophote_list[0]
    sma, step = first_isophote.sample.geometry.reset_sma(step)

    while True:
        sample = EllipseSample(image, sma, geometry=geometry, integrmode=integrmode, sclip=sclip, nclip=nclip)

        sample.update(geometry.fix)
        isophote = Isophote(sample, 0, True, stop_code=4)
        isophote_list.append(isophote)
        sma = isophote.sample.geometry.update_sma(step)
        if minsma and sma <= max(minsma, 0.5):
            break

    isophote_list.sort()
    iso_fix = IsophoteList(isophote_list)
    
    return iso_fix


def align(pa):
    """The function to align the position angle from Si-yue Yu.

    Returns:
        _type_: _description_
    """

    for jc in range(0, len(pa) - 1):
        test = pa[jc + 1] - pa[jc]
        if test > 150:
            pa[jc + 1:] = pa[jc + 1:] - 180
        if test < -150:
            pa[jc + 1:] = pa[jc + 1:] + 180

    for jc in range(0, len(pa) - 1):
        test = pa[jc + 1] - pa[jc]
        if test > 150:
            pa[jc + 1:] = pa[jc + 1:] - 180
        if test < -150:
            pa[jc + 1:] = pa[jc + 1:] + 180

    for jc in range(0, len(pa) - 1):
        test = pa[jc + 1] - pa[jc]
        if test > 150:
            pa[jc + 1:] = pa[jc + 1:] - 180
        if test < -150:
            pa[jc + 1:] = pa[jc + 1:] + 180

    return (pa)


def get_initial_geometry(img, Nsigma=2):
    from Taiji.autoprof_utils import Background_Mode
    
    background = Background_Mode(img)
    bkgsig = background['background noise']

    threshold = np.zeros_like(img)

    threshold[:] = bkgsig * 2

    npixels = 10

    segm = photutils.detect_sources(img, threshold, npixels)

    label = np.argmax(segm.areas) + 1

    segmap = segm.data == label

    raw_segmap = copy.deepcopy(segmap)

    tmp = photutils.SegmentationImage(np.array(segmap, dtype=int))
    tmp_slice = tmp.slices[0]  # you only have one source
    dx = tmp_slice[1].stop - tmp_slice[1].start
    dy = tmp_slice[0].stop - tmp_slice[0].start
    size = max(dx, dy)
    sz = int(0.1 * size)

    segmap_float = ndimage.uniform_filter(np.float64(segmap), size=sz)

    segmap = segmap_float > 0.1
    segmap = np.array(segmap, dtype=int)

    segmap = photutils.SegmentationImage(segmap)

    ##### start to calculate the geometry ##
    s = segmap.slices[0]  # you only have one source
    xmin, xmax = s[1].start, s[1].stop - 1
    ymin, ymax = s[0].start, s[0].stop - 1

    dx = xmax + 1 - xmin
    dy = ymax + 1 - ymin

    xc, yc = xmin + dx // 2, ymin + dy // 2
    dist = int(max(dx, dy) * 1.5 / 2.0)

    ny, nx = img.shape

    slice_stamp = (slice(max(0, yc - dist), min(ny, yc + dist)),
                   slice(max(0, xc - dist), min(nx, xc + dist)))

    xmin_stamp = slice_stamp[1].start
    ymin_stamp = slice_stamp[0].start
    img_stamp = img[slice_stamp]
    segmap_stamp = segmap.data[slice_stamp]

    mask_stamp_bkg = segmap_stamp == 0

    img_stamp_source = np.where(~mask_stamp_bkg, img_stamp,
                                0.0)  # only consider the galaxy segmentation,
    # others are set to zero

    image = np.float64(img_stamp_source)

    # Calculate centroid
    M = skimage.measure.moments(image, order=1)
    yc_stamp = M[1, 0] / M[0, 0]
    xc_stamp = M[0, 1] / M[0, 0]

    xc_centroid = xc_stamp + xmin_stamp
    yc_centroid = yc_stamp + ymin_stamp

    # The covariance matrix of a Gaussian function that has the same second-order moments as the source, with respect to ``(xc, yc)``.

    Mc = skimage.measure.moments_central(image,
                                         center=(yc_stamp, xc_stamp),
                                         order=2)
    assert Mc[0, 0] > 0

    covariance = np.array([[Mc[0, 2], Mc[1, 1]], [Mc[1, 1], Mc[2, 0]]])
    covariance /= Mc[0, 0]  # normalize

    if (covariance[0, 0] <= 0) or (covariance[1, 1] <= 0):
        warnings.warn('Nonpositive second moment.', AstropyUserWarning)
        flag = 1

    rho = 1.0 / 12.0  # variance of 1 pixel-wide top-hat distribution
    x2, xy, xy, y2 = covariance.flat
    while np.abs(x2 * y2 - xy**2) < rho**2:
        x2 += (x2 >= 0) * rho - (x2 < 0) * rho  # np.sign(0) == 0 is no good
        y2 += (y2 >= 0) * rho - (y2 < 0) * rho

    covariance_centroid = np.array([[x2, xy], [xy, y2]])
    eigvals = np.linalg.eigvals(covariance_centroid)
    eigvals_centroid = np.sort(np.abs(eigvals))[::-1]

    a = np.sqrt(np.abs(eigvals_centroid[0]))
    b = np.sqrt(np.abs(eigvals_centroid[1]))

    eps_centroid = 1.0 - (b / a)
    ############# The orientation (in radians) of the source

    x2, xy, xy, y2 = covariance_centroid.flat

    orientation_centroid = 0.5 * np.arctan2(2.0 * xy, x2 - y2)

    pa_centroid = -90 + orientation_centroid * 180. / np.pi

    while pa_centroid < 0:
        pa_centroid += 180
    while pa_centroid > 180:
        pa_centroid -= 180

    results = {}
    results['xc'] = xc_centroid
    results['yc'] = yc_centroid
    results['eps'] = eps_centroid
    results['pa'] = pa_centroid
    results['size'] = max(dx, dy) / 2
    results['rmax'] = max(dx, dy)
    results['segment'] = segmap

    return results

def display_11_residual(data1, data2, xlim_min=0, xlim_max=1, ylim_min=0, ylim_max=1, ylim_res_low=None, ylim_res_high=None, xlabel=None, ylabel=None, kdeplot=False, cmap='mako', bins=50):
    import seaborn as sns
    sns.set_style(None)
    from matplotlib.gridspec import GridSpec
    
    residual = data2-data1
    mean_bias = np.mean(residual)
    #mean_bias = np.sum(mean_bias**2/sn**2 + np.log(sn**2))*(-0.5)
    std_bias = np.std(residual)
    
    # show the scatter plot for original points and residual points in two subplots
    
    one_arr = np.linspace(xlim_min, xlim_max, 99)

    fig = plt.figure(figsize=(6, 8), dpi=80)
    fig.subplots_adjust(left=1,
                        right=2,
                        top=1,
                        bottom=0,
                        wspace=0.01,
                        hspace=0.05)
    gs = GridSpec(ncols=1, nrows=4, figure=fig)

    ax1 = fig.add_subplot(gs[:3, :])

    #plt.errorbar(ell_visual_arr, ellSample_arr, xerr=sn, color='black', fmt='o', lw=1)
    plt.scatter(data1, data2, color='k', s=1)
    if kdeplot:
        sns.histplot(x=data1, y=data2, bins=bins, pthresh=.1, cmap=cmap)
    plt.plot(one_arr, one_arr, color='r', lw=2, label=r'$1:1$', ls='-')
    #plt.plot(one_arr, one_arr+mean_bias, color='green', lw=2, ls='--')
    ax1_text = ax1.text(0.05, 0.95, f'Mean bias $={mean_bias:.3f}$; Std $={std_bias:.3f}$', transform=ax1.transAxes)
    ax1.set_xlim(xlim_min,xlim_max)
    ax1.set_ylim(ylim_min,ylim_max)
    if ylabel is not None:
        plt.ylabel(ylabel, fontsize=18)
    ax1.set_xticklabels([])
    ax1.legend(loc='lower right', fontsize=18)

    ax2 = fig.add_subplot(gs[3:, :])
    plt.scatter(data1, residual, color='k', s=1)
    if kdeplot:
        sns.histplot(x=data1, y=residual, bins=bins, pthresh=.1, cmap=cmap)
    plt.axhline(0, color='r', lw=2)
    plt.ylabel(r'Res.', fontsize=18)
    
    if xlabel is not None:
        plt.xlabel(xlabel, fontsize=18)
    ax2.set_xlim(xlim_min,xlim_max)
    
    if ylim_res_low is not None:
        ax2.set_ylim(ylim_res_low, ylim_res_high)

def add_ellipse(ax, x, y, major_radius, q, pa, color='red', label=None, ls='-', lw=2):
    from matplotlib.patches import Ellipse
    
    ellipse = Ellipse(xy=(x,
                          y),
                      height=major_radius * 2.0,
                      width=major_radius * 2.0 * q,
                      angle=pa)
    ellipse.set_facecolor('none')
    ellipse.set_edgecolor(color)
    ellipse.set_alpha(1)
    ellipse.set_linewidth(lw)
    ellipse.set(clip_box=ax.bbox,
                ls=ls, label=label)
    ax.add_patch(ellipse)
    
def make_lupton_rgb_auto(image_r, image_g, image_b, filename=None):
    
    from astropy.visualization import (AsinhStretch, LinearStretch, LogStretch,
                                       PercentileInterval, PowerStretch,
                                       SinhStretch, SqrtStretch,
                                       ZScaleInterval, make_lupton_rgb)

    stretch = SqrtStretch() + ZScaleInterval(krej=500, contrast=0.05)
    #stretch = SqrtStretch() + PercentileInterval(99.9)

    r = stretch(image_r)
    g = stretch(image_g)
    b = stretch(image_b)

    ### SAVING
    # https://docs.astropy.org/en/stable/api/astropy.visualization.make_lupton_rgb.html
    # astropy.visualization.make_lupton_rgb(image_r, image_g, image_b, minimum=0, stretch=5, Q=8, fil/ename=None)[source]
    # Return a Red/Green/Blue color image from up to 3 images using an asinh stretch.
    # The input images can be int or float, and in any range or bit-depth.

    # Get the value of lower and upper 0.5% of all pixels
    lo_val, up_val = np.percentile(np.hstack((r.flatten(), g.flatten(), b.flatten())), (0.1, 99.9))  
    
    stretch_val = up_val - lo_val
    print(stretch_val)

    rgb_default = make_lupton_rgb(r, g, b, minimum=lo_val, stretch=stretch_val, Q=0, filename=filename)
    
    return rgb_default

def adjust_image_size(image, target_shape, sky_rms):
    from scipy.ndimage import gaussian_filter
    original_shape = image.shape
    padded_shape = target_shape

    if padded_shape[0] > original_shape[0] or padded_shape[1] > original_shape[1]:
        # Compute padding sizes
        height_pad = max(0, padded_shape[0] - original_shape[0])
        width_pad = max(0, padded_shape[1] - original_shape[1])
        
        # Create padded image
        padded_image = np.zeros(padded_shape)
        
        # Compute start indices for embedding original image
        start_row = height_pad // 2
        start_col = width_pad // 2
        
        # Embed original image in padded image
        padded_image[start_row : start_row + original_shape[0], start_col : start_col + original_shape[1]] = image
        
        # Add noise background to the padded image
        noise = np.random.normal(loc=0, scale=sky_rms, size=padded_shape)
        padded_image += noise
        
        return padded_image
    else:
        # Compute crop sizes
        height_crop = (original_shape[0] - padded_shape[0]) // 2
        width_crop = (original_shape[1] - padded_shape[1]) // 2
        
        # Crop image to target shape
        cropped_image = image[height_crop : height_crop + padded_shape[0], width_crop : width_crop + padded_shape[1]]
        
        return cropped_image
    
def _richardson_lucy_np(image, psf, num_iters):
    "Copy from https://github.com/MTLeist255/JWST_Deconvolution/blob/main/Convenience_Functions.py#L85"
    '''11 April 2022: Non-circulant Richardson-Lucy deconvolution algorithm- developed by Brian
    Northan'''
    "This is different from the sckit-image version"
    import scipy.fft as fft
    from astropy.convolution import convolve_fft
    from numpy.fft import fftn, fftshift, ifftn
    
    if image.shape != psf.shape:
        print('padding psf')
        psf,_=pad(psf, image.shape, 'constant')

    otf = fftn(fftshift(psf))
    otf_ = np.conjugate(otf)
    estimate = image
    #estimate = np.ones(image.shape)/image.sum()

    for i in range(num_iters):
        # print(i)
        reblurred = ifftn(fftn(estimate) * otf)
        ratio = image / (reblurred + 1e-30)
        estimate = estimate * (ifftn(fftn(ratio) * otf_)).astype(float)

    return estimate

def pad(img, paddedsize, mode):
    "From https://github.com/True-North-Intelligent-Algorithms/tnia-python/blob/main/tnia/deconvolution/pad.py"
    """ pad image to paddedsize

    Args:
        img ([type]): image to pad 
        paddedsize ([type]): size to pad to 
        mode ([type]): one of the np.pad modes

    Returns:
        [nd array]: padded image
    """
    padding = tuple(map(lambda i,j: ( math.ceil((i-j)/2), math.floor((i-j)/2) ),paddedsize,img.shape))
    return np.pad(img, padding,mode), padding


def unpad(padded, imgsize):
    "From https://github.com/True-North-Intelligent-Algorithms/tnia-python/blob/main/tnia/deconvolution/pad.py"
    """Crop padded back to imgsize.

    Args:
        padded (numpy.ndarray or cupy.ndarray): The padded array.
        imgsize (tuple): The target size of the unpadded array.

    Returns:
        numpy.ndarray or cupy.ndarray: The unpadded array.
    """
    padding = tuple(map(lambda i,j: (math.ceil((i-j)/2), math.floor((i-j)/2)), padded.shape, imgsize))
    slices = tuple(slice(p[0], p[0]+s) for p, s in zip(padding, imgsize))

    return padded[slices]

def richardson_lucy_np(image, psf, num_iters, noncirc=False, mask=None):
    "From https://github.com/True-North-Intelligent-Algorithms/tnia-python/blob/main/tnia/deconvolution/richardson_lucy_np.py"
    """ Deconvolves an image using the Richardson-Lucy algorithm with non-circulant option and option to mask bad pixels, uses numpy

    Note: NumPy FFT functions always cast 32 bit arrays to float64, so passing in 32 bit arrays to save memory will not work. 

    Args:
        image [numpy float array]: the image to be deconvolved 
        psf [numpy float array]: the point spread function
        num_iters (int): the number of iterations to perform
        noncirc (bool, optional): If true use non-circulant edge handling. Defaults to False.
        mask (numpy float array, optional): If not None, use this mask to mask image pixels that should not be considered in the deconvolution. Defaults to None.
            'bad' pixels will be zeroed during the deconvolution and then replaced with the original value after the deconvolution.

    Returns:
        [numpy float array]: the deconvolved image
    """
    
    from numpy.fft import fftn, fftshift, ifftn, ifftshift

    # if noncirc==False and (image.shape != psf.shape) then pad the psf
    if noncirc==False and (image.shape != psf.shape):
        print('padding psf')
        psf,_=pad(psf, image.shape, 'constant')
    
    HTones = np.ones_like(image)

    if (mask is not None):
        HTones = HTones * mask
        mask_values = image*(1-mask)
        image=image*mask
    
    # if noncirc==True then pad the image, psf and HTOnes array to the extended size
    if noncirc:
        # compute the extended size of the image and psf
        extended_size = [image.shape[i]+2*int(psf.shape[i]/2) for i in range(len(image.shape))]

        # pad the image, psf and HTOnes array to the extended size computed above
        original_size = image.shape
        image,_=pad(image, extended_size, 'constant')
        HTones,_=pad(HTones, extended_size, 'constant')
        psf,_=pad(psf, extended_size, 'constant')
    

    otf = fftn(ifftshift(psf))
    otf_ = np.conjugate(otf)

    if noncirc:
        estimate = np.ones_like(image)*np.mean(image)
    else:
        estimate = image

    HTones = np.real(np.fft.ifftn(np.fft.fftn(HTones) * otf_))
    HTones[HTones<1e-6] = 1

    print()
    for i in range(num_iters):
        if i % 10 == 0:
            print(i, end =" ")
        
        reblurred = np.real(ifftn(fftn(estimate) * otf))

        ratio = image / (reblurred + 1e-12)
        correction=np.real((np.fft.ifftn(np.fft.fftn(ratio) * otf_)))
        estimate = estimate * correction/HTones 
    print()        
    
    if noncirc:
        estimate = unpad(estimate, original_size)

    if (mask is not None):
        estimate = estimate*mask + mask_values
    
    return estimate

def Hubble_inclination(q, q_0=0.2):
    cosi = np.sqrt((q**2 - q_0**2) / (1 - q_0**2))
    
    i = np.arccos(cosi) * 180 / np.pi
    
    return i, cosi

if __name__ == '__main__':
    test_pa = -50

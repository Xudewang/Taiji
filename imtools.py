import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
import os
import pandas as pd
import matplotlib as mpl
import matplotlib.image as mpimg
from matplotlib.gridspec import GridSpec
from matplotlib_scalebar.scalebar import ScaleBar
from matplotlib_scalebar.scalebar import ANGLE
from matplotlib import cm
from astropy.visualization.mpl_normalize import ImageNormalize
from astropy.visualization import HistEqStretch, LogStretch
from matplotlib.colors import LogNorm
from astropy.io import ascii
from astropy.modeling.models import custom_model
from astropy.modeling import models, fitting
from astropy.modeling.models import Sersic1D
import re
from uncertainties import unumpy
from astropy.table import Table, Column
import subprocess
import shutil
from scipy.interpolate import interp1d
from matplotlib.patches import Ellipse, Circle
from astropy.stats import bootstrap
from sklearn.utils import resample
from scipy import signal
from scipy.interpolate import interp1d

from matplotlib import colors
from matplotlib.ticker import AutoMinorLocator
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.ticker import NullFormatter
from matplotlib.ticker import MaxNLocator
from astropy.visualization import ZScaleInterval
import matplotlib.patches as mpatches
from matplotlib.patches import Ellipse


def muRe_to_intenRe(muRe, zpt):
    """[summary]

    Args:
        muRe ([type]): [description]
        zpt ([type]): [description]

    Returns:
        [type]: [description]
    """
    intenRe = 10**((zpt - muRe) / 2.5) * 0.259**2
    return intenRe


def all_ninety_pa(pa):
    temp = 0
    if pa > 90:
        temp = pa - 180
    else:
        temp = pa

    return temp


def galaxy_model(x0, y0, PA, ell, I_e, r_e, n):
    model = pyimfit.SimpleModelDescription()
    # define the limits on X0 and Y0 as +/-10 pixels relative to initial values
    model.x0.setValue(x0, [x0 - 10, x0 + 10])
    model.y0.setValue(y0, [y0 - 10, y0 + 10])

    bulge = pyimfit.make_imfit_function('Sersic', label='bulge')
    bulge.I_e.setValue(I_e, [1e-33, 10 * I_e])
    bulge.r_e.setValue(r_e, [1e-33, 10 * r_e])
    bulge.n.setValue(n, [0.5, 5])
    bulge.PA.setValue(PA, [0, 180])
    bulge.ell.setValue(ell, [0, 1])

    model.addFunction(bulge)

    return model


def Ser_kappa(n):
    if n > 0.36:
        bn = 2 * n - 1 / 3 + 4 / (405 * n) + 46 / (25515 * n**2)

    elif n < 0.36:
        bn = 0.01945 - 0.8902 * n + 10.95 * n**2 - 19.67 * n**3 + 13.43 * n**4

    return bn


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


def bright_to_mag(intens, zpt0, texp, pixel_size):
    # for CGS survey, texp = 1s, A = 0.259*0.259
    texp = texp
    A = pixel_size**2
    return -2.5 * np.log10(intens / (texp * A)) + zpt0


def inten_to_mag(intens, zpt0):
    '''
    This function is for calculating the magnitude of total intensity.
    '''
    return -2.5 * np.log10(intens) + zpt0


def inten_tomag_err(inten, err):
    detup_residual = 2.5 * np.log10((inten + err) / inten)
    detdown_residual = 2.5 * np.log10(inten / (inten - err))

    return detup_residual, detdown_residual


def GrowthCurve(sma, ellip, isoInten):
    ellArea = np.pi * ((sma**2.0) * (1.0 - ellip))
    isoFlux = np.append(ellArea[0], [ellArea[1:] - ellArea[:-1]]) * isoInten
    curveOfGrowth = list(
        map(lambda x: np.nansum(isoFlux[0:x + 1]), range(isoFlux.shape[0])))

    indexMax = np.argmax(curveOfGrowth)
    maxIsoSma = sma[indexMax]
    maxIsoFlux = curveOfGrowth[indexMax]

    return np.asarray(curveOfGrowth), maxIsoSma, maxIsoFlux


# firstly, I should judge the file is fits or fit.


def subtract_sky(input_file, sky_value, subtract_sky_file):
    '''
    This function is to subtract the sky value form the original image.
    It is to be modifed, use the easy way to write just the data into the file. Then we do not need consider the header.
    '''

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
    data_list = hdul[0].data

    data_list -= sky_value

    easy_saveData_Tofits(data_list, savefile=subtract_sky_file)


def easy_propagate_err_mu(intens, intens_err):

    return np.array(2.5 / np.log(10) * intens_err / intens)


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


def fix_pa_profile(ellipse_output, pa_col='pa', delta_pa=75.0):
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
    pa = ellipse_output[pa_col].astype('float')

    for i in range(1, len(pa)):
        if (pa[i] - pa[i - 1]) >= delta_pa:
            pa[i] -= 180.0
        elif pa[i] - pa[i - 1] <= (-1.0 * delta_pa):
            pa[i] += 180.0

    ellipse_output[pa_col] = pa

    return ellipse_output


def fix_pa_profile_single(pa_arr, delta_pa=75.0):
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
    from math import floor, ceil

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
    # dPA = 75
    # ellipse_data = fix_pa_profile(ellipse_data, pa_col='pa', delta_pa=dPA)
    # ellipse_data.add_column(
    #     Column(name='pa_norm', data=np.array(
    #         [normalize_angle(pa, lower=-90, upper=90.0, b=True)
    #          for pa in ellipse_data['pa']])))

    # remove the indef
    intens = ellipse_data['intens']
    intens_err = ellipse_data['int_err']
    intens_err_removeindef = removeellipseIndef(intens_err)

    ellipse_data['ell'] = removeellipseIndef(ellipse_data['ell'])
    ellipse_data['pa'] = removeellipseIndef(ellipse_data['pa'])
    ellipse_data['ell_err'] = removeellipseIndef(ellipse_data['ell_err'])
    ellipse_data['pa_err'] = removeellipseIndef(ellipse_data['pa_err'])

    # calculate the magnitude.
    intens_err_removeindef_sky = np.sqrt(
        np.array(intens_err_removeindef)**2 + sky_err**2)
    mu = bright_to_mag(intens - sky_value,
                       zpt0,
                       pixel_size=pixel_size,
                       texp=texp)
    mu_err = easy_propagate_err_mu(
        np.array(intens) - sky_value, intens_err_removeindef_sky)

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


def ellipseGetAvgGeometry(ellipseOut, outRad, minSma=2.0, dPA=75):
    """Get the Average Q and PA."""
    tfluxE = removeellipseIndef(ellipseOut['tflux_e'])
    ringFlux = np.append(tfluxE[0], [tfluxE[1:] - tfluxE[:-1]])

    ellipseOut = fix_pa_profile(ellipseOut, pa_col='pa', delta_pa=dPA)
    ellipseOut.add_column(
        Column(name='pa_norm',
               data=np.array([
                   normalize_angle(pa, lower=-90, upper=90.0, b=True)
                   for pa in ellipseOut['pa']
               ])))

    ell_err = removeellipseIndef(ellipseOut['ell_err'])
    pa_err = removeellipseIndef(ellipseOut['pa_err'])

    try:
        eUse = ellipseOut['ell'][(ellipseOut['sma'] <= outRad)
                                 & (ellipseOut['sma'] >= minSma) &
                                 (np.isfinite(ell_err)) &
                                 (np.isfinite(pa_err))]
        pUse = ellipseOut['pa_norm'][(ellipseOut['sma'] <= outRad)
                                     & (ellipseOut['sma'] >= minSma) &
                                     (np.isfinite(ell_err)) &
                                     (np.isfinite(pa_err))]
        fUse = ringFlux[(ellipseOut['sma'] <= outRad)
                        & (ellipseOut['sma'] >= minSma) &
                        (np.isfinite(ell_err)) & (np.isfinite(pa_err))]
    except Exception:
        try:
            eUse = ellipseOut['ell'][(ellipseOut['sma'] <= outRad)
                                     & (ellipseOut['sma'] >= 0.5) &
                                     (np.isfinite(ell_err)) &
                                     (np.isfinite(pa_err))]
            pUse = ellipseOut['pa_norm'][(ellipseOut['sma'] <= outRad)
                                         & (ellipseOut['sma'] >= 0.5) &
                                         (np.isfinite(ell_err)) &
                                         (np.isfinite(pa_err))]
            fUse = ringFlux[(ellipseOut['sma'] <= outRad)
                            & (ellipseOut['sma'] >= 0.5) &
                            (np.isfinite(ell_err)) & (np.isfinite(pa_err))]
        except Exception:
            eUse = ellipseOut['ell'][(ellipseOut['sma'] <= outRad)
                                     & (ellipseOut['sma'] >= 0.5)]
            pUse = ellipseOut['pa_norm'][(ellipseOut['sma'] <= outRad)
                                         & (ellipseOut['sma'] >= 0.5)]
            fUse = ringFlux[(ellipseOut['sma'] <= outRad)
                            & (ellipseOut['sma'] >= 0.5)]

    avgEll = numpy_weighted_mean(eUse.astype('float'), weights=fUse)
    avgPA = numpy_weighted_mean(pUse.astype('float'), weights=fUse)

    return avgEll, avgPA


def notnan(alist):
    temp = []
    for i in range(len(alist)):
        if not np.isnan(alist[i]):
            temp.append(alist[i])
    return temp


def maxiscal(x1, y1, x2, y2):
    return np.sqrt((x1 - x2)**2 + (y1 - y2)**2)


def bmax(x1, y1, x2, y2):
    return np.sqrt((x1 - x2)**2 + (y1 - y2)**2)


def boxbin(img_sky, nbin):
    '''
    this function is for box bin (smooth).
    '''
    imgR = img_sky.shape[0]
    imgC = img_sky.shape[1]
    nPixelR = int(imgR / nbin)
    nPixelC = int(imgC / nbin)
    imgRb = nPixelR * nbin
    imgCb = nPixelC * nbin

    imgb_sky = img_sky[0:imgRb, 0:imgCb]
    imgBnd_sky = np.zeros([nPixelR, nPixelC])
    for loopR in range(nPixelR):
        for loopC in range(nPixelC):
            R_bin = loopR * nbin
            R_end = (loopR + 1) * nbin
            C_bin = loopC * nbin
            C_end = (loopC + 1) * nbin
            pixel_temp_sky = imgb_sky[R_bin:R_end, C_bin:C_end]

            flat = pixel_temp_sky.flatten()
            flat2 = notnan(flat)
            if len(flat2) == 0:
                imgBnd_sky[loopR, loopC] = np.nan
            else:
                attemp = np.mean(flat2)
                imgBnd_sky[loopR, loopC] = attemp
    return imgBnd_sky


def subtract_source(image, x0, y0, PAs, c, maxis):
    xf1 = x0 - c * np.sin(PAs)
    yf1 = y0 + c * np.cos(PAs)
    xf2 = x0 + c * np.sin(PAs)
    yf2 = y0 - c * np.cos(PAs)

    def distance(x, y):
        return np.sqrt((x - xf1)**2 + (y - yf1)**2) + np.sqrt((x - xf2)**2 +
                                                              (y - yf2)**2)

    sky = np.zeros_like(image)
    for m in range(len(image)):
        for n in range(len(image)):
            if distance(n, m) >= 2 * maxis:
                sky[m][n] = image[m][n]
            else:
                sky[m][n] = np.nan

    return sky


def calculateSky(galaxy_name, maxis=1200):
    galaxy_name = galaxy_name
    imageFile_fit = '/home/dewang/data/CGS/{}/R/{}_R_reg.fit'.format(
        galaxy_name, galaxy_name)
    imageFile_fits = '/home/dewang/data/CGS/{}/R/{}_R_reg.fits'.format(
        galaxy_name, galaxy_name)
    if os.path.exists(imageFile_fit):
        imageFile = imageFile_fit
    elif os.path.exists(imageFile_fits):
        imageFile = imageFile_fits
    data_fits_file = imageFile
    cleandata_fits_file = '/home/dewang/data/CGS/' + \
        galaxy_name + '/R/' + galaxy_name + '_R_reg_clean.fits'
    mask_fits_file = '/home/dewang/data/CGS/' + \
        galaxy_name + '/R/' + galaxy_name + '_R_reg_mm.fits'

    datahdu = fits.open(data_fits_file)
    parafile_from_header = fits.getheader(data_fits_file)
    clean_header = fits.getheader(cleandata_fits_file)
    maskhdu = fits.open(mask_fits_file)
    image = datahdu[0].data
    mask = maskhdu[0].data
    try:
        x0 = parafile_from_header['CEN_X']
        y0 = parafile_from_header['CEN_Y']
        print(x0, y0)
        ellip = parafile_from_header['ell_e']
        PA = parafile_from_header['ell_pa']
    except:
        x0 = clean_header['CEN_X']
        y0 = clean_header['CEN_Y']
        print(x0, y0)
        ellip = clean_header['ell_e']
        PA = clean_header['ell_pa']

    # convert ellipticity to ecentricity/PA, sometimes we should give a e by ourself because that the e/PA of header isnt good enough
    e = np.sqrt((2 - ellip) * ellip)
    #e = 0.3

    #PA = 170
    PAs = PA * np.pi / 180

    maxis = maxis  # 4*parafile_from_header['R80']/0.259
    b = np.sqrt(maxis**2 * (1 - e**2))
    c = maxis * e  # np.sqrt(maxis**2-b**2)
    print(e)
    print(maxis)
    print(PA)

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)

    image_mask_plot = image * ((mask - 1) * (-1))

    norm = ImageNormalize(stretch=HistEqStretch(image))
    #norm = ImageNormalize(stretch=LogStretch(), vmin=0)

    ell1 = Ellipse(xy=(x0, y0),
                   width=2 * b,
                   height=2 * maxis,
                   angle=PA,
                   alpha=0.9,
                   hatch='',
                   fill=False,
                   linestyle='--',
                   color='red',
                   linewidth=1.2)
    ax.imshow(image, origin='lower', norm=norm, cmap='Greys_r')
    ax.add_patch(ell1)

    plt.axis('scaled')
    plt.axis(
        'equal'
    )  # changes limits of x or y axis so that equal increments of x and y have the same length
    # plt.axis('off')
    plt.show()

    # imagemm = np.zeros_like(image)
    # for m in range(len(image)):
    #     for n in range(len(image)):
    #         if mask[m][n] == 0:
    #             imagemm[m][n] = image[m][n] * 1
    #         else:
    #             #imagemm[m][n] = image[m][n]*0
    #             imagemm[m][n] = np.nan

    image[mask > 0] = np.nan

    imagesky = subtract_source(image, x0, y0, PAs, c, maxis)

    # boxsize = 20, calculated by ZhaoYu.
    imagesky_bin = boxbin(imagesky, 20)

    sky_flat_bin = imagesky_bin.flatten()
    sky_flat = imagesky.flatten()

    # sky value and its error
    skyval = np.nanmean(sky_flat)

    skyerr = np.nanstd(sky_flat)
    skyerr_bin = np.nanstd(sky_flat_bin)

    print('sky value: ', skyval)
    print('sky error with box smooth: ', skyerr_bin)
    print('sky error w/o box smooth: ', skyerr)

    return [skyval, skyerr_bin]


def Remove_file(file):
    if os.path.exists(file):
        os.remove(file)


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
    plt.ylabel(r'$\mu_R\ (\mathrm{mag\ arcsec^{-2}})$', fontsize=20)
    plt.xlabel(r'$r\,(\mathrm{arcsec})$', fontsize=20)


def getOuterBound(ellipse_data, sky_err, alter=0.2):
    sma = ellipse_data['sma']
    intens = ellipse_data['intens']
    mu = ellipse_data['mu']
    mu_err = ellipse_data['mu_err']

    mu_err_justsky = easy_propagate_err_mu(intens, sky_err)

    index = mu_err_justsky <= alter

    return sma[index][-1]

def getBound(sma, intens, int_err, zpt0, pixel_size = 0.259, texp=1, alter=0.2):
    mu = bright_to_mag(intens=intens, zpt0=zpt0, texp=texp, pixel_size=pixel_size)
    mu_err = easy_propagate_err_mu(intens = intens, intens_err=int_err)

    index = mu_err <= alter

    return np.array([sma[index][0], sma[index][-1]], dtype=float)


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
               label=''):
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
                    label=label)

    elif plot_style == 'fill':
        ax.plot(sma * pixel_size, ellip, color=color, lw=3, label=label)
        ax.fill_between(sma * pixel_size,
                        ellip + ellip_err,
                        ellip - ellip_err,
                        color=color,
                        alpha=0.5)

    if ylimax:
        ax.set_ylim(ylimin, ylimax)
    else:
        ax.set_ylim(np.nanmin(ellip) - 0.05, np.nanmax(ellip) + 0.05)

    if xlimax:
        ax.set_xlim(xlimin, xlimax)

    ax.set_ylabel(r'Ellipticity')
    ax.set_xlabel(r'$r\,(\mathrm{arcsec})$')
    ax.legend()


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
            label=''):
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
                    label=label)

    elif plot_style == 'fill':
        ax.plot(sma * pixel_size, pa, color=color, lw=3, label=label)
        ax.fill_between(sma * pixel_size,
                        pa + pa_err,
                        pa - pa_err,
                        color=color,
                        alpha=0.5)

    if ylimax:
        ax.set_ylim(ylimin, ylimax)
    else:
        ax.set_ylim(np.nanmin(pa) - 5, np.nanmax(pa) + 5)

    if xlimax:
        ax.set_xlim(xlimin, xlimax)


#     else:
#         plt.xlim(sma[-1]*0.02*(-1)*pixel_size, (sma[-1]+sma[-1]*0.02)*pixel_size)

    ax.set_ylabel(r'PA\, (deg)')
    ax.set_xlabel(r'$r\,(\mathrm{arcsec})$')
    ax.legend()


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
             label=''):
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
                    label=label)

    elif plot_style == 'fill':
        ax.plot(sma * pixel_size, mu, color=color, lw=3, label=label)
        ax.fill_between(sma * pixel_size,
                        mu + mu_err,
                        mu - mu_err,
                        color=color,
                        alpha=0.5)

    if ylimax:
        ax.set_ylim(ylimin, ylimax)
    else:
        ax.set_ylim(np.nanmin(mu) - 0.5, np.nanmax(mu) + 0.5)

    if xlimax:
        ax.set_xlim(xlimin, xlimax)

    ax.legend()
    ax.set_ylabel(r'$\Sigma_R\ (\mathrm{mag\ arcsec^{-2}})$')
    ax.set_xlabel(r'$r\,(\mathrm{arcsec})$')
    plt.gca().invert_yaxis()


def plot_completeSBP(sma,
                     ell,
                     ell_err,
                     pa,
                     pa_err,
                     mu,
                     mu_err,
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

    ax1 = fig.add_subplot(gs[:5, 0])
    plot_ellip(sma,
               ell,
               ell_err,
               pixel_size=pixel_size,
               plot_style=plot_style,
               color=color,
               ylimin=ylimin_e,
               ylimax=ylimax_e,
               xlimin=xlimin,
               xlimax=xlimax)

    ax2 = fig.add_subplot(gs[5:10, 0])
    plot_pa(sma,
            pa,
            pa_err,
            pixel_size=pixel_size,
            plot_style=plot_style,
            color=color,
            ylimin=ylimin_pa,
            ylimax=ylimax_pa,
            xlimin=xlimin,
            xlimax=xlimax)

    ax3 = fig.add_subplot(gs[10:, 0])
    plot_SBP(sma,
             mu,
             mu_err,
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


def LSBImage(ax, dat, noise, pixel_size=0.259, bar_length=50, box_alpha=1):
    ax.imshow(dat,
              origin='lower',
              cmap='Greys',
              norm=ImageNormalize(stretch=HistEqStretch(dat)))
    my_cmap = cm.Greys_r
    my_cmap.set_under('k', alpha=0)
    ax.imshow(np.clip(dat, a_min=noise, a_max=None),
              origin='lower',
              cmap=my_cmap,
              norm=ImageNormalize(stretch=LogStretch(), clip=False),
              clim=[3 * noise, None],
              vmin=3 * noise)
    scalebar = ScaleBar(pixel_size,
                        "''",
                        dimension=ANGLE,
                        color='black',
                        box_alpha= box_alpha,
                        font_properties={'size': 25},
                        location='lower left',
                        length_fraction = pixel_size,
                        fixed_value=bar_length)
    plt.gca().add_artist(scalebar)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.subplots_adjust(left=0.03, right=0.97, top=0.97, bottom=0.05)


# About the Colormaps
IMG_CMAP = plt.get_cmap('viridis')
IMG_CMAP.set_bad(color='black')
SEG_CMAP = random_cmap(ncolors=512, background_color=u'white')
SEG_CMAP.set_bad(color='white')
SEG_CMAP.set_under(color='white')

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


def display_single(img,
                   pixel_scale=0.168,
                   physical_scale=None,
                   xsize=8,
                   ysize=8,
                   ax=None,
                   stretch='arcsinh',
                   scale='zscale',
                   contrast=0.25,
                   zmin=None,
                   zmax=None,
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

    # Stretch option
    if stretch.strip() == 'arcsinh':
        img_scale = np.arcsinh(img)
    elif stretch.strip() == 'log':
        if no_negative:
            img[img <= 0.0] = 1.0E-10
        img_scale = np.log(img)
    elif stretch.strip() == 'log10':
        if no_negative:
            img[img <= 0.0] = 1.0E-10
        img_scale = np.log10(img)
    elif stretch.strip() == 'linear':
        img_scale = img
    else:
        raise Exception("# Wrong stretch option.")

    # Scale option
    if zmin is not None and zmax is not None:
        zmin, zmax = zmin, zmax
    elif scale.strip() == 'zscale':
        try:
            zmin, zmax = ZScaleInterval(
                contrast=contrast).get_limits(img_scale)
        except IndexError:
            # TODO: Deal with problematic image
            zmin, zmax = -1.0, 1.0
    elif scale.strip() == 'percentile':
        try:
            zmin, zmax = AsymmetricPercentileInterval(
                lower_percentile=lower_percentile,
                upper_percentile=upper_percentile).get_limits(img_scale)
        except IndexError:
            # TODO: Deal with problematic image
            zmin, zmax = -1.0, 1.0
    else:
        zmin, zmax = np.nanmin(img_scale), np.nanmax(img_scale)

    show = ax1.imshow(img_scale,
                      origin='lower',
                      cmap=cmap,
                      interpolation='none',
                      vmin=zmin,
                      vmax=zmax)

    # Hide ticks and tick labels
    ax1.tick_params(labelbottom=False,
                    labelleft=False,
                    axis=u'both',
                    which=u'both',
                    length=0)

    # Put scale bar on the image
    (img_size_x, img_size_y) = img.shape
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


def display_isophote(img, x0, y0, sma, ell, pa, ax, pixel_size=0.259):
    """Visualize the isophotes."""

    display_single(img,
                   ax=ax,
                   scale_bar=True,
                   pixel_scale=pixel_size,
                   cmap='Greys_r',
                   scale_bar_length=1)

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

    for k in range(len(sma)):
        if np.logical_and(k % 5 == 0, k > 200):
            e = Ellipse(xy=(x0, y0),
                        height=sma[k] * 2.0,
                        width=sma[k] * 2.0 * (1.0 - ell[k]),
                        angle=pa[k])
            e.set_facecolor('none')
            e.set_edgecolor('#30E3CA')
            e.set_alpha(1)
            e.set_linewidth(2)
            e.set_linestyle('-')
            ax.add_artist(e)

    for k in range(len(sma)):
        if np.logical_and(k % 15 == 0, k <= 200):
            e = Ellipse(xy=(x0, y0),
                        height=sma[k] * 2.0,
                        width=sma[k] * 2.0 * (1.0 - ell[k]),
                        angle=pa[k])
            e.set_facecolor('none')
            e.set_edgecolor('#30E3CA')
            e.set_alpha(1)
            e.set_linewidth(1)
            e.set_linestyle('-')
            ax.add_artist(e)


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
    """Visualize the isophotes."""

    display_single(img,
                   ax=ax,
                   scale_bar=True,
                   pixel_scale=pixel_size,
                   cmap='Greys_r',
                   scale_bar_length=scale_bar_length)

    LSBImage(ax=ax,
             dat=img,
             noise=noise,
             pixel_size=pixel_size,
             bar_length=scale_bar_length)

    for k in range(len(sma)):
        if k % 2 == 0:
            e = Ellipse(xy=(x0, y0),
                        height=sma[k] * 2.0,
                        width=sma[k] * 2.0 * (1.0 - ell[k]),
                        angle=pa[k])
            e.set_facecolor('none')
            e.set_edgecolor('#FF2E63')
            e.set_alpha(1)
            e.set_linewidth(1.5)
            ax.add_artist(e)

    for k in range(len(sma)):
        if np.logical_and(k % 5 == 0, k > 200):
            e = Ellipse(xy=(x0, y0),
                        height=sma[k] * 2.0,
                        width=sma[k] * 2.0 * (1.0 - ell[k]),
                        angle=pa[k])
            e.set_facecolor('none')
            # e.set_edgecolor('#30E3CA')
            e.set_edgecolor('magenta')
            e.set_alpha(1)
            e.set_linewidth(2)
            e.set_linestyle('-')
            ax.add_artist(e)

    for k in range(len(sma)):
        if np.logical_and(k % 15 == 0, k <= 200):
            e = Ellipse(xy=(x0, y0),
                        height=sma[k] * 2.0,
                        width=sma[k] * 2.0 * (1.0 - ell[k]),
                        angle=pa[k])
            e.set_facecolor('none')
            e.set_edgecolor('#08D9D6')
            e.set_alpha(1)
            e.set_linewidth(1)
            e.set_linestyle('-')
            ax.add_artist(e)


def easy_saveData_Tofits(data, savefile):
    hdu = fits.PrimaryHDU(data)

    hdul = fits.HDUList(hdu)
    hdul.writeto(savefile, overwrite=True)

def exptime_modify(data, exptime, savefile, opper='divide'):
    if opper == 'divide':
        data/=exptime
    elif opper == 'multiply':
        data*=exptime
    
    easy_saveData_Tofits(data, savefile=savefile)
    print(opper+' exposure time. Finished!')

def get_bulge_geo_galfit_input(input_file):

    with open(input_file) as f:
        input_data = f.read()

    mue_t = re.search('(?<=3\)\s).*(?=#\s\sSurface)', input_data)[0]
    mue = re.search('.*(?=\s[0-9])', mue_t)[0]
    print('mue = ', mue)

    Re_t = re.search('(?<=4\)\s).*(?=#\s\sR_e)', input_data)[0]
    Re = re.search('.*(?=\s[0-9])', Re_t)[0]
    print('Re = ', Re)

    sersicn_t = re.search('(?<=5\)\s).*(?=#\s\sSersic)', input_data)[0]
    sersicn = re.search('.*(?=\s[0-9])', sersicn_t)[0]
    print('sersic index = ', sersicn)

    sky_value_t = re.search('(?<=1\)\s).*(?=#\s\sSky)', input_data)[0]
    sky_value = re.search('.*(?=\s[0-9])', sky_value_t)[0]
    print('sky value = ', sky_value)

    return np.array([mue, Re, sersicn, sky_value], dtype=str)

def get_disk_geo_galfit_output(input_file):
    '''
    input: the Galfit input/output file.

    return: ellipticity and position angle. data_type: float value of a numpy array.
    '''


    with open(input_file) as f:
        input_data = f.read()

    disk_geo_data = re.search('(?<=0\)\sexpdisk).*(?=#\s\sPosition)', input_data, re.DOTALL)[0]
    #print(disk_geo_data)
    axisratio_disk_data = re.search('(?<=9\)\s).*(?=#\s\sAxis)', disk_geo_data)[0]
    axisratio_disk_galfit = float(re.search('.*(?=\s[0-9])', axisratio_disk_data)[0])
    e_disk_galfit = 1 - axisratio_disk_galfit
    pa_disk_data = re.search('(?<=10\)).*', disk_geo_data)[0]
    pa_disk_galfit = float(re.search('.*(?=\s[0-9])', pa_disk_data)[0])

    # print('galfit disk ell of {0}  = '.format(galaxy_name), e_disk_galfit)
    # print('galfit disk PA of {0} = '.format(galaxy_name), pa_disk_galfit)

    return np.array([e_disk_galfit, pa_disk_galfit], dtype=float)


if __name__ == '__main__':
    test_pa = -50

    lll = normalize_angle(test_pa, 0, 180)

    print(lll)

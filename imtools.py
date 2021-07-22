import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
import os
from pyraf import iraf
import pandas as pd
import matplotlib as mpl
import matplotlib.image as mpimg
from matplotlib.gridspec import GridSpec
from matplotlib_scalebar.scalebar import ScaleBar
from matplotlib_scalebar.scalebar import ANGLE
from astropy.visualization.mpl_normalize import ImageNormalize
from astropy.visualization import HistEqStretch, LogStretch
from matplotlib.colors import LogNorm
from astropy.io import ascii
from astropy.modeling.models import custom_model
from astropy.modeling import models,fitting
from astropy.modeling.models import Sersic1D
import re
from uncertainties import unumpy
from astropy.table import Table,Column
import subprocess
import shutil
from scipy.interpolate import interp1d
from matplotlib.patches import Ellipse, Circle
from astropy.stats import bootstrap
from sklearn.utils import resample 
from scipy import signal

iraf.stsdas()
iraf.analysis()
iraf.isophote()

def muRe_to_intenRe(muRe, zpt):
    intenRe = 10**((zpt-muRe)/2.5)*0.259**2
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
    bulge.I_e.setValue(I_e, [1e-33, 10*I_e])
    bulge.r_e.setValue(r_e, [1e-33, 10*r_e])
    bulge.n.setValue(n, [0.5, 5])
    bulge.PA.setValue(PA, [0, 180])
    bulge.ell.setValue(ell, [0, 1])

    model.addFunction(bulge)

    return model

def Ser_kappa(n):
    if n > 0.36:
        bn = 2*n - 1/3 + 4/(405*n) + 46/(25515*n**2)
        
    elif n < 0.36:
        bn = 0.01945 - 0.8902*n + 10.95 * n**2 - 19.67 * n **3 + 13.43 * n **4
        
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
                temp.append(pa[i]+180)
            else:
                temp.append(pa[i])
        temp = np.array(temp)

    return temp

def bright_to_mag(intens, zpt0, texp):
    # for CGS survey, texp = 1s, A = 0.259*0.259
    texp = 1
    A = 0.259**2
    return -2.5 * np.log10(intens / (texp * A)) + zpt0

def inten_to_mag(intens, zpt0):
    '''
    This function is for calculating the magnitude of total intensity.
    '''
    return -2.5 * np.log10(intens) + zpt0

def inten_tomag_err(inten,err):
    detup_residual = 2.5 * np.log10((inten + err) / inten)
    detdown_residual = 2.5 * np.log10(inten / (inten - err))
    
    return detup_residual,detdown_residual

def GrowthCurve(sma, ellip, isoInten):
    ellArea = np.pi * ((sma ** 2.0) * (1.0 - ellip))
    isoFlux = np.append(ellArea[0], [ellArea[1:] - ellArea[:-1]]) * isoInten
    curveOfGrowth = list(map(lambda x: np.nansum(isoFlux[0:x + 1]), range(isoFlux.shape[0])))
    
    indexMax = np.argmax(curveOfGrowth)
    maxIsoSma = sma[indexMax]
    maxIsoFlux = curveOfGrowth[indexMax]

    return np.asarray(curveOfGrowth), maxIsoSma, maxIsoFlux

def Remove_file(file):
    if os.path.exists(file):
        os.remove(file)

# firstly, I should judge the file is fits or fit.

def maskFitsTool(inputImg_fits_file, mask_fits_file):
    
    # firstly, I should judge the file is fits or fit.
    if inputImg_fits_file[-3:] == 'its':
        mask_pl_file = inputImg_fits_file.replace('.fits','.fits.pl')
    elif inputImg_fits_file[-3:] == 'fit':
        mask_pl_file = inputImg_fits_file.replace('.fit','.pl')

    if os.path.exists(mask_pl_file):
        print('pl file exists')
    else:
        print('pl file does not exist and we should make a pl file for ellipse task')
        iraf.imcopy(mask_fits_file, mask_pl_file)

def subtract_sky(input_file, mask_file, sky_value):
    modfile = input_file.replace('.fit','_sky.fit')
    Remove_file(modfile)

    hdul = fits.open(input_file)
    header_list = hdul[0].header
    data_list = hdul[0].data

    data_list -= sky_value
    try:
        header_list.rename_keyword('D26.5C','D26_5C')
        header_list.rename_keyword('D26.5', 'D26_5')
    except ValueError as e:
        print(e)
        print("no D26.5")

    hdul.writeto(modfile) 
    
    maskFitsTool(modfile, mask_file)
    
def propagate_err_mu(intens, intens_err, zpt0, pix = 0.259 , exp_time = 1):
    '''    
    How to propagate the error.
    
    ellipse_data: the data array from Iraf ellipse task.
    
    zpt0: zero point of the magnitude
    
    pix: pixel scale
    
    exp_time: exposure time
    
    '''
    texp = exp_time
    A = pix**2
    
#     intens = ellipse_data['intens']
#     intens_err = ellipse_data['int_err']
    
#     intens_err_removeindef = removeellipseIndef(intens_err)
    
#     intens_err[intens_err=='INDEF'] = np.nan 
    
#     intens_err_removeindef = [float(intens_err[i]) for i in range(len(intens_err))]

    uncertainty_inten = unumpy.uarray(list(intens),list(intens_err))

    uncertainty_mu = -2.5 * unumpy.log10(uncertainty_inten / (texp * A)) + zpt0
    uncertainty_mu_value = unumpy.nominal_values(uncertainty_mu)
    uncertainty_mu_std = unumpy.std_devs(uncertainty_mu)

    return uncertainty_mu_value, uncertainty_mu_std

def easy_propagate_err_mu(intens, intens_err):
    
    return np.array(2.5/np.log(10)*intens_err/intens)

def removeellipseIndef(arr):
    
    # let the indef equals NaN
    arr[arr=='INDEF'] = np.nan
    
    # convert the str array into float array
    
    arr_new = [float(arr[i]) for i in range(len(arr))]
    
    return np.array(arr_new)

def imageMax(image_data_temp, mask_data_temp):

    # combine the image and mask
    image_data_temp[mask_data_temp>0]=np.nan
    image_value_max = np.nanmax(image_data_temp)

    return image_value_max

def ellipseGetGrowthCurve(ellipOut,useTflux=False):
    """
    Extract growth curve from Ellipse output.
    Parameters: ellipOut
    Parameters: useTflux, if use the cog of iraf or not.
    """
    if not useTflux:
        # The area in unit of pixels covered by an elliptical isophote
        ell = removeellipseIndef(ellipOut['ell'])
        ellArea = np.pi * ((ellipOut['sma'] ** 2.0) * (1.0 - ell))
        # The total flux inside the "ring"
        intensUse = ellipOut['intens']
        isoFlux = np.append(ellArea[0], [ellArea[1:] - ellArea[:-1]]) * ellipOut['intens']
        # Get the growth Curve
        curveOfGrowth = list(map(lambda x: np.nansum(isoFlux[0:x + 1]), range(isoFlux.shape[0])))
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
    

def PyrafEllipse(input_img, outTab, outDat, cdf, pf, inisma, maxsma, x0, y0,
              pa, ell_e, zpt0, interactive = False, inellip='', hcenter=False, hpa=False, hellip=False, 
              nclip=3, usclip=3, lsclip=2.5, FracBad=0.9, olthresh=0, intemode='median',step=0.1, sky_err=0, maxgerr=0.5, harmonics=False):
    
    if not os.path.isfile(input_img):
        raise Exception("### Can not find the input image: %s !" % input_img)
    
    if os.path.exists(outTab):
        os.remove(outTab)
    if os.path.exists(outDat):
        os.remove(outDat)
    if os.path.exists(cdf):
        os.remove(cdf)
    if os.path.exists(pf):
        os.remove(pf)

    iraf.ellipse(interactive = interactive,input = input_img, out=outTab,fflag= FracBad, sma0 = inisma, 
                 maxsma=maxsma,x0=x0,nclip=nclip, usclip=usclip,lsclip=lsclip,
             y0=y0,pa = pa, e = ell_e, hcenter=hcenter,hpa = hpa,hellip = hellip,olthresh=olthresh,
                 integrmode=intemode, step=step, inellip=inellip, maxgerr=maxgerr, harmonics=harmonics)

    iraf.tdump(table=outTab, datafile=outDat, cdfile=cdf, pfile=pf)
    
    print('The ellipse finished!')
    
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
    
    # calculate the magnitude.
    intens_err_removeindef_sky = np.sqrt(np.array(intens_err_removeindef)**2 + sky_err**2)
    mu = bright_to_mag(intens, zpt0)
    mu_err = easy_propagate_err_mu(np.array(intens), intens_err_removeindef_sky)
    
    ellipse_data.add_column(Column(name='mu', data=mu ))
    ellipse_data.add_column(Column(name='mu_err', data = mu_err))
    
    return ellipse_data

def readEllipse(outDat, zpt0, sky_err):
    
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
    
    # calculate the magnitude.
    intens_err_removeindef_sky = np.sqrt(np.array(intens_err_removeindef)**2 + sky_err**2)
    mu = bright_to_mag(intens, zpt0)
    mu_err = easy_propagate_err_mu(np.array(intens), intens_err_removeindef_sky)
    
    ellipse_data.add_column(Column(name='mu', data=mu ))
    ellipse_data.add_column(Column(name='mu_err', data = mu_err))
    

    return ellipse_data

def readGalfitInput(input_file):
    with open(input_file) as f:
        input_data = f.read()
        
    mue = re.search('(?<=3\)\s).*(?=\s[0-9])', input_data)[0]    
    Re = re.search('(?<=4\)\s).*(?=\s[0-9])', input_data)[0]
    n = re.search('(?<=5\)\s).*(?=\s[0-9])', input_data)[0]
    
    sky_value_t = re.search('(?<=1\)\s).*(?=#\s\sSky)', input_data)[0]
    sky_value = re.search('.*(?=\s[0-9])', sky_value_t)[0]
    
    return np.array([mue,Re,n,sky_value])

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
        Column(name='pa_norm', data=np.array(
            [normalize_angle(pa, lower=-90, upper=90.0, b=True)
             for pa in ellipseOut['pa']])))

    ell_err = removeellipseIndef(ellipseOut['ell_err'])
    pa_err = removeellipseIndef(ellipseOut['pa_err'])
    
    try:
        eUse = ellipseOut['ell'][(ellipseOut['sma'] <= outRad) &
                                 (ellipseOut['sma'] >= minSma) &
                                 (np.isfinite(ell_err)) &
                                 (np.isfinite(pa_err))]
        pUse = ellipseOut['pa_norm'][(ellipseOut['sma'] <= outRad) &
                                     (ellipseOut['sma'] >= minSma) &
                                     (np.isfinite(ell_err)) &
                                     (np.isfinite(pa_err))]
        fUse = ringFlux[(ellipseOut['sma'] <= outRad) &
                        (ellipseOut['sma'] >= minSma) &
                        (np.isfinite(ell_err)) &
                        (np.isfinite(pa_err))]
    except Exception:
        try:
            eUse = ellipseOut['ell'][(ellipseOut['sma'] <= outRad) &
                                     (ellipseOut['sma'] >= 0.5) &
                                     (np.isfinite(ell_err)) &
                                     (np.isfinite(pa_err))]
            pUse = ellipseOut['pa_norm'][(ellipseOut['sma'] <= outRad) &
                                         (ellipseOut['sma'] >= 0.5) &
                                         (np.isfinite(ell_err)) &
                                         (np.isfinite(pa_err))]
            fUse = ringFlux[(ellipseOut['sma'] <= outRad) &
                            (ellipseOut['sma'] >= 0.5) &
                            (np.isfinite(ell_err)) &
                            (np.isfinite(pa_err))]
        except Exception:
            eUse = ellipseOut['ell'][(ellipseOut['sma'] <= outRad) &
                                     (ellipseOut['sma'] >= 0.5)]
            pUse = ellipseOut['pa_norm'][(ellipseOut['sma'] <= outRad) &
                                         (ellipseOut['sma'] >= 0.5)]
            fUse = ringFlux[(ellipseOut['sma'] <= outRad) &
                            (ellipseOut['sma'] >= 0.5)]

    avgEll = numpy_weighted_mean(eUse.astype('float'), weights=fUse)
    avgPA = numpy_weighted_mean(pUse.astype('float'), weights=fUse)

    return avgEll, avgPA

def notnan(alist):
    temp = []
    for i in range(len(alist)):
        if not np.isnan(alist[i]):
            temp.append(alist[i])
    return temp

def maxiscal(x1,y1,x2,y2):
    return np.sqrt((x1-x2)**2 + (y1-y2)**2)

def bmax(x1,y1,x2,y2):
    return np.sqrt((x1-x2)**2 + (y1-y2)**2)

def boxbin(img_sky,nbin):
    '''
    this function is for box bin (smooth).
    '''
    imgR = img_sky.shape[0]
    imgC = img_sky.shape[1]
    nPixelR = int(imgR/nbin)
    nPixelC = int(imgC/nbin)
    imgRb = nPixelR * nbin
    imgCb = nPixelC * nbin

    imgb_sky = img_sky[0:imgRb, 0:imgCb]
    imgBnd_sky = np.zeros([nPixelR, nPixelC])
    for loopR in range(nPixelR):
        for loopC in range(nPixelC):
            R_bin = loopR * nbin
            R_end = (loopR+1) * nbin
            C_bin = loopC * nbin
            C_end = (loopC+1) * nbin
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
    xf1 = x0 - c*np.sin(PAs)
    yf1 = y0 + c*np.cos(PAs)
    xf2 = x0 + c*np.sin(PAs)
    yf2 = y0 - c*np.cos(PAs)

    def distance(x,y):
        return np.sqrt((x-xf1)**2 + (y-yf1)**2) + np.sqrt((x-xf2)**2 + (y-yf2)**2)

    sky = np.zeros_like(image)
    for m in range(len(image)):
        for n in range(len(image)):
            if distance(n,m) >= 2*maxis:
                sky[m][n] = image[m][n]
            else:
                sky[m][n] = np.nan
                
    return sky

def calculateSky(galaxy_name, maxis = 1200):
    galaxy_name = galaxy_name
    imageFile_fit = '/home/dewang/data/CGS/{}/R/{}_R_reg.fit'.format(galaxy_name, galaxy_name)
    imageFile_fits = '/home/dewang/data/CGS/{}/R/{}_R_reg.fits'.format(galaxy_name, galaxy_name)
    if os.path.exists(imageFile_fit):
        imageFile = imageFile_fit
    elif os.path.exists(imageFile_fits):
        imageFile = imageFile_fits
    data_fits_file = imageFile
    cleandata_fits_file = '/home/dewang/data/CGS/'+galaxy_name+'/R/'+galaxy_name+'_R_reg_clean.fits'
    mask_fits_file = '/home/dewang/data/CGS/'+galaxy_name+'/R/'+galaxy_name+'_R_reg_mm.fits'

    datahdu = fits.open(data_fits_file)
    parafile_from_header = fits.getheader(data_fits_file)
    clean_header = fits.getheader(cleandata_fits_file)
    maskhdu = fits.open(mask_fits_file)
    image = datahdu[0].data
    mask = maskhdu[0].data
    try:
        x0 = parafile_from_header['CEN_X']
        y0 = parafile_from_header['CEN_Y']
        print(x0,y0)
        ellip = parafile_from_header['ell_e']
        PA = parafile_from_header['ell_pa']
    except:
        x0 = clean_header['CEN_X']
        y0 = clean_header['CEN_Y']
        print(x0,y0)
        ellip = clean_header['ell_e']
        PA = clean_header['ell_pa']

    # convert ellipticity to ecentricity/PA, sometimes we should give a e by ourself because that the e/PA of header isnt good enough 
    e = np.sqrt((2-ellip)*ellip)
    #e = 0.3
   
    #PA = 170
    PAs = PA * np.pi/180

    maxis = maxis#4*parafile_from_header['R80']/0.259
    b = np.sqrt(maxis**2*(1-e**2))
    c = maxis*e #np.sqrt(maxis**2-b**2)
    print(e)
    print(maxis)
    print(PA)     

    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(111)

    image_mask_plot = image*((mask-1)*(-1))

    norm = ImageNormalize(stretch=HistEqStretch(image))
    #norm = ImageNormalize(stretch=LogStretch(), vmin=0)

    ell1 = Ellipse(xy = (x0, y0), width = 2*b, height = 2*maxis, 
                angle = PA, alpha=0.9,hatch='',fill=False,linestyle='--',color='red',linewidth=1.2)
    ax.imshow(image,origin='lower', norm=norm, cmap='Greys_r')
    ax.add_patch(ell1)

    plt.axis('scaled')
    plt.axis('equal')   #changes limits of x or y axis so that equal increments of x and y have the same length
    #plt.axis('off')
    plt.show()           

    # imagemm = np.zeros_like(image)
    # for m in range(len(image)):
    #     for n in range(len(image)):
    #         if mask[m][n] == 0:
    #             imagemm[m][n] = image[m][n] * 1
    #         else:
    #             #imagemm[m][n] = image[m][n]*0
    #             imagemm[m][n] = np.nan 

    image[mask>0] = np.nan
                        
    imagesky = subtract_source(image, x0, y0, PAs, c, maxis)

    # boxsize = 20, calculated by ZhaoYu.
    imagesky_bin = boxbin(imagesky,20)

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
        
def round_up_to_odd(f):
    f = int(np.ceil(f))
    return f - 1 if f % 2 == 0 else f

# I want to calculate the scale length using finite diff.
def get_local_h(r, sbp):
    mu_obs = sbp
    local_h_arr = []
    for i in range(len(r)):
        if i-2<0:
            temp_r = r[0:i+3]
            temp_mu = mu_obs[0:i+3]
        else:
            temp_r = r[i-2:i+3]
            temp_mu = mu_obs[i-2:i+3]
        
        #print(temp_r,temp_mu)

        # calculate the deviation using finite difference
        deltayx = (temp_mu[-1]+temp_mu[-2] - temp_mu[1] - temp_mu[0])/(6*1)
        
        #print(m)
        h_local_temp = 1.086/deltayx
        local_h_arr.append(h_local_temp)
        #print(h_local_temp)
    return local_h_arr

def medfil_h(local_h, frac_rad):
    kernel_size = round_up_to_odd(frac_rad*local_h.size)
    
    return signal.medfilt(local_h, kernel_size)

def cs(h_arr):
    h_mean = np.mean(h_arr)
    cs0 = 0
     
    cs_arr = []
    cs_arr.append(cs0)
    cs_temp = cs0
    for i in range(len(h_arr)):
        h_diff = h_arr[i] - h_mean
        cs_temp += h_diff
        cs_arr.append(cs_temp)
    
    cs_min = np.min(cs_arr)
    cs_min_loca = np.argmin(cs_arr)
    cs_max = np.max(cs_arr)
    cs_max_loca = np.argmax(cs_arr)

    cs_diff = cs_max - cs_min 

    return (cs_arr, cs_min_loca, cs_min, cs_max_loca, cs_max, cs_diff)


def cs_bootstrap(h_arr, bootstrap_size):
    origi_cs_diff = cs(h_arr)[-1]
    bootresult = bootstrap(h_arr, bootstrap_size)
    cs_diff_arr = [cs(bootresult[i])[-1] for i in range(len(bootresult))]
    cs_diff_arr = np.array(cs_diff_arr)
    cs_diff_small = cs_diff_arr[cs_diff_arr < origi_cs_diff]

    confidence = len(cs_diff_small)/bootstrap_size

    return (confidence, origi_cs_diff, cs_diff_arr)

 
def cs_bootstrap_woreplace(h_arr, bootstrap_size, replace = True):
    origi_cs_diff = cs(h_arr)[-1]
    bootresult = np.array([resample(h_arr, n_samples=len(h_arr), replace=replace) for i in range(bootstrap_size)])

    cs_diff_arr = [cs(bootresult[i])[-1] for i in range(len(bootresult))]
    cs_diff_arr = np.array(cs_diff_arr)
    cs_diff_small = cs_diff_arr[cs_diff_arr < origi_cs_diff]

    confidence = len(cs_diff_small)/bootstrap_size

    return (confidence, origi_cs_diff, cs_diff_arr)

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
    
    plt.plot(sma_sky*pixel_size, mu_sky)
    plt.fill_between(sma_sky*pixel_size, mu_sky - mu_err_sky, mu_sky+mu_err_sky, alpha=0.2)
    plt.ylim(np.min(mu_sky)-0.2, np.max(mu_sky)+0.2)
    plt.gca().invert_yaxis()
    plt.ylabel(r'$\mu_R\ (\mathrm{mag\ arcsec^{-2}})$', fontsize=20)
    plt.xlabel(r'$r\,(\mathrm{arcsec})$', fontsize=20)

if __name__ == '__main__':
    test_pa = -50

    lll = normalize_angle(test_pa, 0, 180)

    print('normlize pa:', lll)

    xxx = calculateSky('NGC6754')
    print(xxx)






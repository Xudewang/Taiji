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

from matplotlib import colors
from matplotlib.ticker import AutoMinorLocator
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.ticker import NullFormatter
from matplotlib.ticker import MaxNLocator
from astropy.visualization import ZScaleInterval
import matplotlib.patches as mpatches
from matplotlib.patches import Ellipse


def round_up_to_odd(f):
    f = int(np.ceil(f))
    return f - 1 if f % 2 == 0 else f


def medfil_h_old(local_h, frac_rad=0.1):
    kernel_size = round_up_to_odd(frac_rad * local_h.size)

    return signal.medfilt(local_h, kernel_size)


def median_kernel_size_old(h_arr, length_h, frac):
    kernel_size_temp = int(length_h * frac)

    if (kernel_size_temp % 2) == 0:
        kernel_size = kernel_size_temp - 1

    else:
        kernel_size = kernel_size_temp

    return kernel_size


def median_kernel_size(length_h, frac):
    '''
    This function is to get the median size of the smoothing. Basically it should be a odd number and considering the fraction.
    '''
    kernel_size_temp = length_h * frac

    kernel_size = round_up_to_odd(kernel_size_temp)

    return kernel_size


def median_smooth_h(local_h_arr, length_h, frac):
    '''
    This is a simple function to perform the median filter to the local scale length profile.
    ----------------------
    Parameters:
    local_h_arr: the array should be median smoothed.

    length_h: the length of the local scale length array, but should note that this value should be a physical and real observation value not the the number of array with interpolation.

    frac: what fraction of the length to give the kernel size.
    '''
    kernel_size = median_kernel_size(length_h=length_h, frac=frac)

    return signal.medfilt(local_h_arr, kernel_size)


def get_local_h_2order(r, mu_obs):
    local_h_arr_fd = []
    #h_err_2order = []
    dx = r[1] - r[0]
    for i in range(len(r)):
        if i - 1 < 0:

            deltayx = (-3 * mu_obs[i] + 4 * mu_obs[i + 1] -
                       mu_obs[i + 2]) / (2 * dx)
            #err_temp = np.sqrt(9/4/dx**2*mu_err[i]**2+16/4/dx**2*mu_err[i+1]**2+1/4/dx**2*mu_err[i+2]**2)

        elif i - (len(r) - 1) > -1:

            deltayx = (3 * mu_obs[i] - 4 * mu_obs[i - 1] + mu_obs[i - 2])
            #err_temp = np.sqrt(9/4/dx**2*mu_err[i]**2+16/4/dx**2*mu_err[i-1]**2+1/4/dx**2*mu_err[i-2]**2)
        else:

            # calculate the deviation using finite difference
            #deltayx = (-mu_obs[i+2]+8*mu_obs[i+1]-8*mu_obs[i-1]+mu_obs[i-2])/(12*dx)
            deltayx = (mu_obs[i + 1] - mu_obs[i - 1]) / (2 * dx)

            #err_temp = np.sqrt(1/4/dx**2*mu_err[i+1]**2+1/4/dx**2*mu_err[i-1]**2)

        # print(m)
        h_local_temp = 1.086 / deltayx
        local_h_arr_fd.append(h_local_temp)
        # h_err_2order.append(err_temp)

    return np.array(local_h_arr_fd)


def Get_localh_withmedian_old(sma, mu, length_h, frac):

    f = interp1d(sma, mu)

    xnew = np.arange(np.ceil(np.min(sma)), np.floor(np.max(sma)), 1)

    ynew = f(xnew)

    local_h = get_local_h_2order(xnew, ynew)
    local_h_medfil = median_smooth_h(local_h, length_h=length_h, frac=frac)

    return np.array([xnew, ynew, local_h, local_h_medfil])

def Get_localh_withmedian(sma, mu, step=1, frac=0.1):
    """This function is to get the local scale length, then median smooth the local h profile.

    Args:
        sma (numpy array): the input radial radius. Units: pixel
        mu ([type]): [description]
        step (float): the step of interpolation. Unit: pixel, it is 1 pixel by default.
        frac ([type]): [description]

    Returns:
        [type]: [description]
    """

    f = interp1d(sma, mu)

    xnew = np.arange(np.ceil(np.min(sma)), np.floor(np.max(sma)), step)

    ynew = f(xnew)

    local_h = get_local_h_2order(xnew, ynew)

    # to get the length of xnew, finally I think the kernel size should use this length, because this actually the radial length of new profile.
    length_h = xnew[-1] - xnew[0]
    local_h_medfil = median_smooth_h(local_h, length_h=length_h, frac=frac)

    return np.array([xnew, ynew, local_h, local_h_medfil])


def cs(h_arr):
    #TODO: combine this cs function and find_max function.
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

    confidence = len(cs_diff_small) / bootstrap_size

    return (confidence, origi_cs_diff, cs_diff_arr)


def cs_bootstrap_woreplace(h_arr, bootstrap_size, replace=True):
    origi_cs_diff = cs(h_arr)[-1]
    bootresult = np.array([
        resample(h_arr, n_samples=len(h_arr), replace=replace)
        for i in range(bootstrap_size)
    ])

    cs_diff_arr = [cs(bootresult[i])[-1] for i in range(len(bootresult))]
    cs_diff_arr = np.array(cs_diff_arr)
    cs_diff_small = cs_diff_arr[cs_diff_arr < origi_cs_diff]

    confidence = len(cs_diff_small) / bootstrap_size

    return (confidence, origi_cs_diff, cs_diff_arr)


def cs_bootstrap_savecs(h_arr, bootstrap_size):
    origi_cs_diff = cs(h_arr)[-1]
    bootresult = bootstrap(h_arr, bootstrap_size)
    cs_diff_arr = [cs(bootresult[i])[-1] for i in range(len(bootresult))]
    cs_diff_arr = np.array(cs_diff_arr)
    cs_diff_small = cs_diff_arr[cs_diff_arr < origi_cs_diff]

    confidence = len(cs_diff_small)/bootstrap_size

    cs_boot_arr = [cs(bootresult[i])[0] for i in range(len(bootresult))]

    return (confidence, origi_cs_diff, cs_diff_arr, cs_boot_arr)


def find_max(sma, cs_arr):
    """
    This function is to find the maximum of the cs.

    input: the results of cs function.

    output: the location of the maximum. And this maximum represents the change point, i.e., the localtion of disk break.

    """
    cs_arr = cs_arr

    cs_abs_arr = np.abs(cs_arr)
    cs_abs_max = np.max(cs_abs_arr)
    cs_max_loca = np.argmax(cs_abs_arr)

    return sma[cs_max_loca]


def find_sigma(hprofile_ori, hprofile, rb, R, p1, p2, savefile=''):
    rplus1 = rb + int(p1*R)
    rplus2 = rb + int(p2*R)

    rminus1 = rb - int(p1*R)
    rminus2 = rb - int(p2*R)

    maxr2 = np.max(hprofile[rminus2:rplus2])
    minr2 = np.min(hprofile[rminus2:rplus2])

    deltah = np.abs(maxr2 - minr2)
    sigmaleft = np.std(hprofile_ori[rminus2:rminus1])
    sigmaright = np.std(hprofile_ori[rplus1:rplus2])

    plt.figure()
    plt.plot(r, hprofile_ori, 'o',
             label='Local scale length derived by finite difference', color='black')
    plt.plot(r, hprofile, label='Local scale length after median filter', color='red')
    plt.axvline(rplus1, color='blue', ls='-.',
                label=r'$5\%$ and $10\%$ apart from the break radius', alpha=0.5)
    plt.axvline(rplus2, color='blue', ls='-.', alpha=0.5)
    plt.axvline(rminus1, color='blue', ls='-.', alpha=0.5)
    plt.axvline(rminus2, color='blue', ls='-.', alpha=0.5)
    plt.xlabel('R', fontsize=30)
    plt.ylabel(r'$h_\mathrm{local}$', fontsize=30)
    plt.ylim(6, 35)
    plt.axvline(rb, color='gray', ls='--', lw=3, label='Disk break')
    plt.legend()

    if savefile:
        plt.savefig(
            savefile)
            
    plt.show()

    return np.array([deltah, sigmaleft, sigmaright])


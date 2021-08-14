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

        #print(m)
        h_local_temp = 1.086 / deltayx
        local_h_arr_fd.append(h_local_temp)
        #h_err_2order.append(err_temp)

    return np.array(local_h_arr_fd)


def Get_localh_withmedian(sma, mu, length_h, frac):

    f = interp1d(sma, mu)

    xnew = np.arange(np.ceil(np.min(sma)), np.floor(np.max(sma)), 1)

    ynew = f(xnew)

    local_h = get_local_h_2order(xnew, ynew)
    local_h_medfil = median_smooth_h(local_h, length_h=length_h, frac=frac)

    return np.array([xnew, local_h, local_h_medfil])

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

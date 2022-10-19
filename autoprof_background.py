from scipy.stats import iqr, norm
from scipy.optimize import minimize
import numpy as np
import copy


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
        "background uncertainty": uncertainty
    }

def My_Background_Mode(IMG, mask=None):
    # TODO: wrong, should modify.
    # Mask main body of image so only outer 1/5th is used
    # for background calculation.
    if mask is not None:
        IMG_copy = copy.deepcopy(IMG)
        IMG_copy[mask==1] = np.nan
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

    return {
        "background": bkgrnd,
        "background noise": noise
    }

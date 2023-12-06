# functions written by Si-Yue Yu (phyyueyu@gmail.com)

import numpy as np
import os
import scipy.integrate as integrate
from scipy import interpolate
from astropy.cosmology import FlatLambdaCDM
from astropy.io import fits
import copy
import subprocess
import pyimfit
from extract_fix_isophotes import extract_fix_isophotes
from dustmaps.sfd import SFDQuery
from astropy.coordinates import SkyCoord
from astropy.modeling import models, fitting
from SYU_intensity_profiles import cart2pol, reducePA
import warnings
from astropy.utils.exceptions import AstropyUserWarning
import photutils
import scipy.ndimage as ndi
import time
import statmorph
from frebin import frebin
import contextlib
from photutils.psf import create_matching_kernel
from astropy.convolution import convolve
from autoprof_background import Background_Mode
import skimage.measure


#def get_psf_moffat_imfit(fwhm):
#    p='/Users/syu/works/MissSpirals/code/ImgSimulation/input/psf/moffat_psf.conf'
#    config_file = np.loadtxt(p, dtype=str, delimiter='@')
#    config_file[8] = 'fwhm  '+str(fwhm)
#    np.savetxt('moffat_psf.conf', config_file, fmt='%-70s')
#    bashCommand = '/Users/syu/Software/Imfit/imfit-1.8.0/makeimage moffat_psf.conf -o psf_imfit.fits'
#    print('----- running makeimage to generate a binned local psf ------')
#    process = subprocess.run(bashCommand.split())
#    moff = fits.getdata('psf_imfit.fits')
#    print('----- end of makeimage ------')
#    return moff 

def get_psf_moffat_imfit(fwhm, beta=3):
    # print("get psf model from imfit".center(50, "-"))
    model = pyimfit.SimpleModelDescription()
    model.x0.setValue(33, fixed=True)
    model.y0.setValue(33, fixed=True)
    func = pyimfit.make_imfit_function("Moffat", label="psf")
    func.I_0.setValue(0.0043)
    func.PA.setValue(0, fixed=True)
    func.ell.setValue(0, fixed=True)
    func.fwhm.setValue(fwhm)
    func.beta.setValue(beta, fixed=True)
    model.addFunction(func)
    myfit = pyimfit.Imfit(model)

    with contextlib.redirect_stdout(None):
        psf = myfit.getModelImage(shape=(65,65))
    psf = psf / np.sum(psf)

    return psf


def read_noise_information():
    # only support f200w
    noise_info = {'ff_electrons':17778.0, 'unsat':7.884202520074018, 'rn':10.96, 'nramps':3, 
                       'nexp':3, 'excessp1':1, 'excessp2':5, 'm':8, 
                       'tframe':10.73677, 'tgroup':107.3677}

    return noise_info


def get_jwst_throughput(wavelength, fil):
    # dir_pandeia_jwst = '/Users/syu/Software/pandeia_data-1.7/jwst/'
    # path_ote_thruput = dir_pandeia_jwst + 'telescope/jwst_telescope_ote_thruput.fits'
    # path_internal_thruput = dir_pandeia_jwst + 'nircam/optical/jwst_nircam_internaloptics_throughput_20170216143448.fits'
    # path_dichroic_thruput = dir_pandeia_jwst + 'nircam/optical/jwst_nircam_sw_dbs_20160923153727.fits'
    # path_QE = dir_pandeia_jwst + 'nircam/qe/jwst_nircam_sw_qe_20160902164019.fits'
    # trans_f200w = dir_pandeia_jwst + 'nircam/filters/jwst_nircam_f200w_trans_20160902164019.fits'
    # trans_f150w = dir_pandeia_jwst + 'nircam/filters/jwst_nircam_f150w_trans_20160902164019.fits'
    # trans = {'f200w':trans_f200w, 'f150w':trans_f150w}

    # ote_thruput = fits.getdata(path_ote_thruput)
    # ote_eff = np.interp(wavelength, ote_thruput['wavelength'], ote_thruput['throughput'])
    
    # filter_thruput = fits.getdata(trans[fil])
    # filter_eff = np.interp(wavelength, filter_thruput['wavelength'], filter_thruput['throughput'])
    
    # base_thruput = fits.getdata(path_internal_thruput)
    # base_eff = np.interp(wavelength, base_thruput['wavelength'], base_thruput['throughput'])
    # dichroic_thruput = fits.getdata(path_dichroic_thruput)
    # dichroic_eff = np.interp(wavelength, dichroic_thruput['wavelength'], dichroic_thruput['throughput'])
    # internal_eff = base_eff * dichroic_eff
    
    # _QE_ = fits.getdata(path_QE)
    # QE = np.interp(wavelength, _QE_['wavelength'], _QE_['throughput'])

    # througput = ote_eff * filter_eff * internal_eff * QE
    fil = fil.upper()
    dir_T = '/Users/syu/Software/My_Python/SGA_JWST/input/nircam_throughputs/modAB_mean/nrc_plus_ote/'
    path_T = dir_T + fil + '_NRC_and_OTE_ModAB_mean.txt'
    T = np.loadtxt(path_T, skiprows=1)
    Tx = T[:,0]
    Ty = T[:,1]
    througput = np.interp(wavelength, Tx, Ty)

    return througput

def find_close_filter(mywave, allbpass, allwave, alldata):
    delt = abs(allwave-mywave)
    idd = np.argmin(delt)
    return allwave[idd], allbpass[idd], alldata[idd]


def get_close_filter(w, fils):
    bandpass = read_bandpass()
    wcents = []
    for fil in fils:
        wcents.append(bandpass[fil]['Wpivot'])
    delt = abs(w - wcents)
    idd = np.argmin(delt)

    return fils[idd]



def cal_pivot_wavelength(band):
    return np.sqrt(integrate.simps(band[:,1]*band[:,0], band[:,0])/integrate.simps(band[:,1]/band[:,0], band[:,0]))

def read_bandpass(path_fil=None):
    filter_names = ['fuv', 'nuv', 'u', 'g', 'r', 'i', 'z', 'f814w', 'f160w', 'f115w', 'f150w', 'f200w', 'f356w', 'f444w']
    allfilters = ['galex_FUV.pb', 'galex_NUV.pb', 'sdss_up.pb', 'sdss_gp.pb', 'sdss_rp.pb', 'sdss_ip.pb', 
                  'sdss_zp.pb', 'hst_F814W.pb', 'hst_F160W.pb', 'F115W.dat', 'F150W.dat', 
                  'F200W.dat', 'F356W.dat', 'F444W.dat']
    path_fil = '/Users/syu/Software/My_Python/SGA_JWST/input/filters' if path_fil is None else path_fil

    results = {}
    for j in range(len(allfilters)):
        band = np.loadtxt( os.path.join(path_fil, allfilters[j]) )
        Wcent = cal_pivot_wavelength(band)
        results[filter_names[j]] = {'Wpivot':Wcent, 'Trans':band}

    return results

# def size_evolution(lgmass):
#     xarr = np.array([ 9.25,  9.5 ,  9.75, 10.  , 10.25, 10.5 , 10.75, 11.  , 11.25])
#     # yarr = np.array([-0.417, -0.498, -0.53 , -0.543, -0.542, -0.593, -0.687, -0.796, -0.895])
#     yarr = np.array([-0.417, -0.496, -0.525, -0.532, -0.522, -0.563, -0.663, -0.761, -0.845])
#     fsize = interpolate.interp1d(xarr, yarr, kind='linear', fill_value="extrapolate")
#     return fsize(lgmass)*1.0

def vanderWel_size_evo(lgmass, Type='late'):
    if Type == 'late':
        Marray_disk = np.array([9.25, 9.75, 10.25, 10.75, 11.25])
        Beta_disk = np.array([-0.48, -0.63, -0.52, -0.72, -0.80])
        if (lgmass < min(Marray_disk)) or (lgmass > max(Marray_disk)):
            return np.nan
        
        pms = np.polyfit(Marray_disk, Beta_disk, 2)
        f = np.poly1d(pms)
    if Type == 'early':
        Marray_E = np.array([9.75, 10.25, 10.75, 11.25])
        Beta_E = np.array([-0.22, -1.01, -1.24, -1.32])
        if (lgmass < min(Marray_E)) or (lgmass > max(Marray_E)):
            return np.nan
        
        pms = np.polyfit(Marray_E, Beta_E, 3)
        f = np.poly1d(pms)
    
    return f(lgmass)

# def SB_evo_with_size_evo_subtracted(mym, myw):
#     xarr = np.array([ 9.25,  9.5 ,  9.75, 10.  , 10.25, 10.5 , 10.75, 11.  , 11.25])
#     yarrs = \
#     [np.array([-2.29509091, -2.36312912, -2.29357041, -2.14692739, -1.79396148,\
#             -1.3281514 , -0.51810454,  0.04035939,  0.42275285]),
#      np.array([-2.18367407, -2.24120971, -2.2291468 , -2.07910769, -1.74835374,\
#             -1.28851543, -0.49373997,  0.01064529,  0.44314358]),
#      np.array([-2.076792  , -2.13092213, -2.14722626, -2.01650618, -1.70021484,\
#             -1.30558316, -0.5151808 , -0.11908739,  0.09190567]),
#      np.array([-1.97088041, -2.01831728, -2.04075832, -1.90366823, -1.57236017,\
#             -1.20205728, -0.4881286 , -0.20249097, -0.03242844]),
#      np.array([-1.79422004, -1.82534063, -1.84219304, -1.64163746, -1.33413898,\
#             -1.02920412, -0.52160937, -0.3694988 , -0.24262503]),
#      np.array([-1.70147171, -1.71097111, -1.71623884, -1.53888048, -1.22666025,\
#             -0.96084827, -0.60513594, -0.43651775, -0.35484819]),
#      np.array([-1.41218279, -1.40123716, -1.43888911, -1.23619437, -0.95338833,\
#             -0.72882939, -0.46834183, -0.35867103, -0.29167956]),
#      np.array([-1.29168412, -1.28415878, -1.32353766, -1.1460035 , -0.88568788,\
#             -0.69614725, -0.51342488, -0.39882431, -0.34365555]),
#      np.array([-1.22264308, -1.18324984, -1.22243779, -1.0605733 , -0.84516576,\
#             -0.69632373, -0.54437979, -0.47694688, -0.44102377]),
#      np.array([-1.05699383, -1.00358018, -1.05134837, -0.8842991 , -0.69835377,\
#             -0.61748084, -0.55447911, -0.58072687, -0.58100167])]

#     waves = np.array([1400, 1700, 2200, 2700, 3605.07, 4413.08, 5512.10, 6594.7, 8059.8, 12411.8])
#     xfit = mym
#     yfit_store = np.array([])
#     for jk in range(len(yarrs)):
#         yarr = yarrs[jk]

#         fmu = interpolate.interp1d(xarr, yarr, kind='linear', fill_value="extrapolate")
#         yfit = fmu(xfit)
#         yfit_store = np.append(yfit_store, yfit)

#     fmu = interpolate.interp1d(waves, yfit_store, kind='linear', fill_value="extrapolate")
    
#     return fmu(myw).item()

def Abs_mag_evo(lgmass, wave, late=True, early=False):
    if (late == True) and (early == False):
        mass_s = np.array([ 9.25,  9.75, 10.25, 10.75, 11.25,  9.25,  9.75, 10.25, 10.75,
           11.25,  9.25,  9.75, 10.25, 10.75, 11.25,  9.25,  9.75, 10.25,
           10.75, 11.25,  9.25,  9.75, 10.25, 10.75, 11.25,  9.25,  9.75,
           10.25, 10.75, 11.25])
        lambda_s = np.array([ 3598.45600801,  3598.45600801,  3598.45600801,  3598.45600801,
            3598.45600801,  4398.1665325 ,  4398.1665325 ,  4398.1665325 ,
            4398.1665325 ,  4398.1665325 ,  5500.38195704,  5500.38195704,
            5500.38195704,  5500.38195704,  5500.38195704,  6562.42606116,
            6562.42606116,  6562.42606116,  6562.42606116,  6562.42606116,
            8046.48758626,  8046.48758626,  8046.48758626,  8046.48758626,
            8046.48758626, 12382.65005999, 12382.65005999, 12382.65005999,
           12382.65005999, 12382.65005999])
        alpha_s = np.array([1.39, 1.76, 1.64, 0.97, 0.81, 1.21, 1.54, 1.37, 0.77, 0.73, 1.04,
            1.25, 1.11, 0.59, 0.64, 0.93, 1.1 , 1.  , 0.61, 0.7 , 0.71, 0.84,
            0.86, 0.57, 0.56, 0.53, 0.5 , 0.6 , 0.57, 0.63 ])
        if (lgmass < min(mass_s)) or (lgmass > max(mass_s)) or (wave < min(lambda_s)) or (wave > max(lambda_s)):
            return np.nan
        lglambda_s = np.log10(lambda_s)

        p_init = models.Polynomial2D(degree=2, fixed={'c2_0':True, 'c0_2':False})
        fit_p = fitting.LevMarLSQFitter()

        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', message='Model is linear in parameters',
                            category=AstropyUserWarning)
            polyfit = fit_p(p_init, mass_s, lglambda_s, alpha_s)
        
    elif early == True:
        mass_s = np.array([ 9.75, 10.25, 10.75, 11.25,  9.75, 10.25, 10.75, 11.25,  9.75,
           10.25, 10.75, 11.25,  9.75, 10.25, 10.75, 11.25,  9.75, 10.25,
           10.75, 11.25,  9.75, 10.25, 10.75, 11.25])
        lambda_s = np.array([ 3598.45600801,  3598.45600801,  3598.45600801,  3598.45600801,
            4398.1665325 ,  4398.1665325 ,  4398.1665325 ,  4398.1665325 ,
            5500.38195704,  5500.38195704,  5500.38195704,  5500.38195704,
            6562.42606116,  6562.42606116,  6562.42606116,  6562.42606116,
            8046.48758626,  8046.48758626,  8046.48758626,  8046.48758626,
           12382.65005999, 12382.65005999, 12382.65005999, 12382.65005999])
        alpha_s = np.array([1.13, 1.22, 1.02, 1.25, 0.98, 1.06, 0.88, 1.12, 0.89, 0.86, 0.7 ,
            0.93, 0.79, 0.77, 0.58, 0.77, 0.63, 0.6 , 0.48, 0.71, 0.49, 0.44,
            0.31, 0.65])
        if (lgmass < min(mass_s)) or (lgmass > max(mass_s)) or (wave < min(lambda_s)) or (wave > max(lambda_s)):
            return np.nan
        lglambda_s = np.log10(lambda_s)

        p_init = models.Polynomial2D(degree=2, fixed={'c2_0':True, 'c0_2':False})
        fit_p = fitting.LevMarLSQFitter()

        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', message='Model is linear in parameters',
                            category=AstropyUserWarning)
            polyfit = fit_p(p_init, mass_s, lglambda_s, alpha_s)
    
    return polyfit(lgmass, np.log10(wave))
    

def dlumi2redshift(d_lumi):
    cosmo = FlatLambdaCDM(H0=70, Om0=0.27)
    z_arr = np.arange(0+0.0001, 10, 0.001)
    dlumi_arr = cosmo.luminosity_distance(z_arr)
    f_tmp = interpolate.interp1d(dlumi_arr, z_arr)
    z = f_tmp(d_lumi)
    return z.item()

def dlumi2dang(d_lumi, z):
    return d_lumi/(1+z)**2

def correct_flux_zeropoint(img, old=None, new=None):
    return img * 10**(0.4*(new-old))

def find_hst_zeropoint(band):
    store = {'f606w':26.49, 'f814w':25.94, 'f850lp':24.84, 'f275w':24.14, 'f336w':24.64, 
             'f350lp':26.94, 'f105w':26.27, 'f125w':26.25, 'f160w':25.96}
    return store[band.lower()]

def rms2exptimeF814W(rms):
#	print('--Transfomation of rms to exptime **only** support F814W --')
    k, b = -0.52877944, -0.50269368
    return int(10**((np.log10(rms)-b)/k))

def exptime2rmsF814W(t):
#	print('--Transfomation of exptime to rms **only** support F814W --')
    k, b = -0.52877944, -0.50269368
    return 10**(k*np.log10(t)+b)

def rms2exptimeF160W(rms):
    k, b = -0.48175206, -0.37548644
    return int(10**((np.log10(rms)-b)/k))

def exptime2rmsF160W(t):
    k, b = -0.48175206, -0.37548644
    return 10**(k*np.log10(t)+b)

def gaussian(height, center_x, center_y, width_x, width_y):
	"""Returns a gaussian function with the given parameters"""
	width_x = float(width_x)
	width_y = float(width_y)
	return lambda x,y: height*np.exp(
			-(((center_x-x)/width_x)**2+((center_y-y)/width_y)**2)/2)



def make_same_DIM_with(mat, shape):
    if np.shape(mat)[0] <= shape[0]:
        x_mat = int(np.shape(mat)[0]/2); y_mat = int(np.shape(mat)[1]/2)
        zeros = np.zeros(shape)
        x_nx = int(shape[0]/2); y_ny = int(shape[1]/2)
    
        ix = 0 if np.mod(np.shape(mat)[0], 2) == 0 else 1
        iy = 0 if np.mod(np.shape(mat)[1], 2) == 0 else 1
        zeros[x_nx-x_mat:x_nx+x_mat+ix, y_ny-y_mat:y_ny+y_mat+iy] = mat
    else:
        x_nx = int(shape[0]/2); y_ny = int(shape[1]/2)
        x_mat = int(np.shape(mat)[0]/2); y_mat = int(np.shape(mat)[1]/2)
        zeros = copy.deepcopy(mat[x_mat-x_nx:x_mat+x_nx+1, y_mat-y_ny:y_mat+y_ny+1])

    return zeros


def check_FWHM(path_psfs):
    psf_g = fits.getdata(path_psfs['g'])
    psf_r = fits.getdata(path_psfs['r'])
    psf_z = fits.getdata(path_psfs['z'])

    xcen = psf_r.shape[0]//2
    ycen = psf_r.shape[0]//2

    iso_g = extract_fix_isophotes(image=psf_g, xcen=xcen, ycen=ycen, eps=0.0, pa=0, step=0.1, initsma = 10, silent=True)
    f = interpolate.interp1d(iso_g.intens/np.max(psf_g), iso_g.sma)
    fwhm_g = 2*f(0.5)
    ############################
    iso_r = extract_fix_isophotes(image=psf_r, xcen=xcen, ycen=ycen, eps=0.0, pa=0, step=0.1, initsma = 10, silent=True)
    f = interpolate.interp1d(iso_r.intens/np.max(psf_r), iso_r.sma)
    fwhm_r = 2*f(0.5)
    ############################
    iso_z = extract_fix_isophotes(image=psf_z, xcen=xcen, ycen=ycen, eps=0.0, pa=0, step=0.1, initsma = 10, silent=True)
    f = interpolate.interp1d(iso_z.intens/np.max(psf_z), iso_z.sma)
    fwhm_z = 2*f(0.5)
    ############################

    sequence = np.array(['g', 'r', 'z'])
    fwhms = np.array([fwhm_g, fwhm_r, fwhm_z])

    indx = np.argsort(-1*fwhms)

    sequence = sequence[indx]
    fwhms = fwhms[indx]

    fwhms_dict = {sequence[0]:fwhms[0], sequence[1]:fwhms[1], sequence[2]:fwhms[2]}

    return sequence, fwhms_dict



def rank_FWHM(path_psfs):
    psf_g = fits.getdata(path_psfs['g'])
    psf_r = fits.getdata(path_psfs['r'])
    psf_z = fits.getdata(path_psfs['z'])

    fits.writeto('tmp_g.fits', psf_g, overwrite=True)
    fits.writeto('tmp_r.fits', psf_r, overwrite=True)
    fits.writeto('tmp_z.fits', psf_z, overwrite=True)

    xc = psf_r.shape[0]//2 + 1
    yc = psf_r.shape[0]//2 + 1

    write_moffat_conf(x0=xc, y0=yc, I_0=np.max(psf_g), fix_beta=False)
    bashCommand = '/Users/syu/Software/Imfit/imfit-1.8.0/imfit tmp_g.fits -c tmp_moffat_imfit.conf --sky=1'
    process = subprocess.run(bashCommand.split())
    best = read_best_fit()
    fwhm_g = best['fwhm']

    write_moffat_conf(x0=xc, y0=yc, I_0=np.max(psf_r), fix_beta=False)
    bashCommand = '/Users/syu/Software/Imfit/imfit-1.8.0/imfit tmp_r.fits -c tmp_moffat_imfit.conf --sky=1'
    process = subprocess.run(bashCommand.split())
    best = read_best_fit()
    fwhm_r = best['fwhm']

    write_moffat_conf(x0=xc, y0=yc, I_0=np.max(psf_z), fix_beta=False)
    bashCommand = '/Users/syu/Software/Imfit/imfit-1.8.0/imfit tmp_z.fits -c tmp_moffat_imfit.conf --sky=1'
    process = subprocess.run(bashCommand.split())
    best = read_best_fit()
    fwhm_z = best['fwhm']
    ############################

    sequence = np.array(['g', 'r', 'z'])
    fwhms = np.array([fwhm_g, fwhm_r, fwhm_z])

    indx = np.argsort(-1*fwhms)

    sequence = sequence[indx]
    fwhms = fwhms[indx]

    fwhms_dict = {sequence[0]:fwhms[0], sequence[1]:fwhms[1], sequence[2]:fwhms[2]}

    return sequence, fwhms_dict


def write_moffat_conf(x0=33, y0=33, PA=0, ell=0, I_0=0.05, fwhm=4, beta=3, fix_beta=False):
    conf = [' ' for i in range(8)]
    conf[0] = 'X0 {} fixed'.format(x0)
    conf[1] = 'Y0 {} fixed'.format(y0)
    conf[2] = 'FUNCTION   Moffat'
    conf[3] = 'PA {} fixed'.format(PA)
    conf[4] = 'ell {} fixed'.format(ell)
    conf[5] = 'I_0 {}'.format(I_0)
    conf[6] = 'fwhm {}'.format(fwhm)
    conf[7] = 'beta {} fixed'.format(beta) if fix_beta else 'beta {} '.format(beta)

    np.savetxt('tmp_moffat_imfit.conf', conf, fmt='%-50s')
    return 

def read_best_fit():
    file = 'bestfit_parameters_imfit.dat'
    result = np.loadtxt(file, dtype=str, delimiter='@')
    best = {}
    best['X0'] = float(result[0][4:])
    best['Y0'] = float(result[1][4:])
    best['PA'] = float(result[3][4:])
    best['ell'] = float(result[4][5:])
    best['I_0'] = float(result[5][5:])
    best['fwhm'] = float(result[6][6:])
    best['beta'] = float(result[7][6:])
    return best


def add_info_to_clean_hdr(hdu, clean):
    hdu.header['ap_bkg'] = clean.ap_bkg
    hdu.header['ap_sig'] = clean.ap_bkg_noise
    hdu.header['ap_BE'] = clean.ap_bkg_err
    hdu.header['bkg'] = clean.bkg
    hdu.header['sig'] = clean.bkg_noise
    hdu.header['BE'] = clean.bkg_err
   
    hdu.header.comments['ap_bkg'] = 'bkg autoprof'
    hdu.header.comments['ap_sig'] = 'bkg noise from autoprof'
    hdu.header.comments['ap_BE'] = 'uncertainty of bkg from autoprof'
    hdu.header.comments['bkg'] = 'bkg by syu'
    hdu.header.comments['sig'] = 'bkg noise by syu'
    hdu.header.comments['BE'] = 'uncertainty of bkg by syu'
    
    return hdu


def get_MW_transmission_rate(lam, ra, dec):
    # https://irsa.ipac.caltech.edu/workspace/TMP_vDl64V_24107/DUST/NGC895.v0001/extinction.html
    LamEff = [0.3734, 0.4309, 0.5517, 0.6520, 0.8007, 0.4621, 0.6546, 
          0.8111, 0.3587, 0.4717, 0.6165, 0.7476, 0.8923, 1.248, 
          1.659, 2.190, 3.52, 4.46]
    Aebv = [4.107, 3.641, 2.682, 2.119, 1.516, 3.381, 2.088, 1.487, 
        4.239, 3.303, 2.285, 1.698, 1.263, 0.709, 0.449, 0.302, 0.178, 0.148]
    f_Aebv = interpolate.interp1d(LamEff, Aebv)

    coord = SkyCoord(ra, dec, unit='deg', frame='icrs')
    sfd = SFDQuery()
    ebv = sfd(coord)

    A = f_Aebv(lam/10000) * ebv 
    rate = 10**(-0.4*A)

    return rate



def dist_ellipse_sma(shape=None, xc=None, yc=None, ab=None , eps=None, PA=None, pa=None):

    ab = (1-eps) if eps is not None else ab
    PA = PA if PA is not None else pa

    nx = shape[1]
    ny = shape[0]

    ang = PA * np.pi / 180.
    cosang = np.cos(ang)
    sinang = np.sin(ang)

    x = np.arange(0, nx) - xc
    y = np.arange(0, ny) - yc

    angle = np.zeros((ny, nx))
    sma = np.zeros((ny, nx))

    xcosang = x*cosang
    xsinang = x*sinang

    for i in range(0, ny):
        xtemp = xcosang + y[i]*sinang
        ytemp = -1.0 * xsinang + y[i]*cosang

        rho, phi = cart2pol(xtemp/ab, ytemp)
        angle[i, :] = phi[:] * 180./np.pi
        sma[i, :] = rho[:]
    return sma

def clean_outer(img, rout, eps, PA):
    xc = img.shape[0]//2
    mat = dist_ellipse_sma(shape=img.shape, xc=xc, yc=xc, eps=eps, PA=PA)
    icu = (mat > rout) & (img > 9)
    img[icu] = 0

    return img



def convert_Es_per_mJy_to_ZP(Es_per_mJy):
    return 2.5*np.log10(Es_per_mJy) + 7.5 + 2.5*np.log10(3631)

def convert_ZP_to_Es_per_mJy(ZP):
    return 10**(0.4*(ZP - 7.5 - 2.5*np.log10(3631)))

def convert_Es_per_mJy_to_MJyperStr_per_Es(Es_per_mJy):
#   pix_to_str = 2.11539874851881E-14 # str per pixel; 0.03**2*(4.848137E-6)**2
    pix_to_str = 0.03**2*(4.848137E-6)**2
    return 1/Es_per_mJy/pix_to_str * 1E-9

def convert_ZP_to_MJyperStr_per_Es(ZP):
#   pix_to_str = 2.11539874851881E-14 # str per pixel
    pix_to_str = 0.03**2*(4.848137E-6)**2
    Es_per_mJy = convert_ZP_to_Es_per_mJy(ZP)
    return convert_Es_per_mJy_to_MJyperStr_per_Es(Es_per_mJy)

def convert_ZP_to_mJy_per_Es(ZP):
    return 1./convert_ZP_to_Es_per_mJy(ZP)


def convert_MJyperStr_per_Es_to_Es_per_mJy(MJyperStr_per_Es):
#   pix_to_str = 2.11539874851881E-14 # str per pixel; 0.03**2*(4.848137E-6)**2
    pix_to_str = 0.03**2*(4.848137E-6)**2
    return 1/MJyperStr_per_Es/pix_to_str * 1E-9

def convert_MJyperStr_per_Es_to_ZP(MJyperStr_per_Es):
    Es_per_mJy = convert_MJyperStr_per_Es_to_Es_per_mJy(MJyperStr_per_Es)
    return convert_Es_per_mJy_to_ZP(Es_per_mJy)


def convert_PHOTMJSR_to_ZP(PHOTMJSR, channel='SW'):
    if channel == 'SW':
        gain = 2.05
    if channel == 'LW':
        gain = 1.82
    return convert_MJyperStr_per_Es_to_ZP(PHOTMJSR/gain)

def myfunc(x, a, n, b, m):
    y = a*x**n + b*x**m
    return y

def convert_nanomaggy_to_10nJy(img):
    ftot = 10 ** ( 0.4 * (16.4 - 22.5) ) * 1e5
    return img * ftot

def convert_10nJy_to_nanomaggy(img):
    ftot = 10 ** ( 0.4 * (16.4 - 22.5) ) * 1e5
    return img / ftot

#def sigma_of_my_DESI_in_10nJy(img, bkgsig=None):
#    params = np.array([4.06499363e-04, 1.35931427e+00, 7.18865235e-04, 1.20771339e+00])
#    tmp = copy.deepcopy(img)
#    tmp[tmp<0]=0
#    var = np.zeros_like(img)
#    var[:] = myfunc(tmp[:], a=params[0], n=params[1], b=params[2], m=params[3])
#    sigma = np.sqrt(var)
#    sigma = np.sqrt(sigma**2 + bkgsig**2)
#    return sigma

# def sigma_of_my_DESI_in_nanomaggy(img, bkgsig=None):
#     params = np.array([4.06499363e-04, 1.35931427e+00, 7.18865235e-04, 1.20771339e+00])
#     img_10nJy = convert_nanomaggy_to_10nJy(img)
#     tmp = copy.deepcopy(img_10nJy)
#     tmp[tmp<0]=0
#     var = np.zeros_like(img)
#     var[:] = myfunc(tmp[:], a=params[0], n=params[1], b=params[2], m=params[3])
#     sigma = np.sqrt(var)
#     sigma_nmaggy = convert_10nJy_to_nanomaggy(sigma)
#     sigma_tot = np.sqrt(sigma_nmaggy**2 + bkgsig**2)

#     return sigma_tot


def sigma_of_my_DESI_in_nanomaggy(img, bkgsig=None):
    '''
    See notebook: /Users/syu/works/MissSpirals/code/pipeline/define_sample/cal_DESI_invariance.ipynb
    '''
    img_10nJy = convert_nanomaggy_to_10nJy(img)
    tmp = copy.deepcopy(img_10nJy)
    tmp[tmp<0]=0
    var = np.zeros_like(img)
    var[:] = tmp[:] * 0.0242
    sigma = np.sqrt(var)
    sigma_nmaggy = convert_10nJy_to_nanomaggy(sigma)
    sigma_tot = np.sqrt(sigma_nmaggy**2 + bkgsig**2)

    return sigma_tot

def sigma_of_my_DESI_in_10nJy(img, bkgsig=None):
    ''' 
    See notebook: /Users/syu/works/MissSpirals/code/pipeline/define_sample/cal_DESI_invariance.ipynb
    '''
    img_10nJy = img
    tmp = copy.deepcopy(img_10nJy)
    tmp[tmp<0]=0
    var = np.zeros_like(img)
    var[:] = tmp[:] * 0.0242
    sigma = np.sqrt(var)
    sigma_10nJy = np.sqrt(sigma**2 + bkgsig**2)

    return sigma_10nJy




def get_noise_array(survey='ceers', fil='f200w'):
    survey = survey.lower()
    fil = fil.lower()

    dir_err = '/Users/syu/works/MissSpirals/code/JWST/'

    file_rate = dir_err + 'redshift_bins.txt'
    rate = np.loadtxt(file_rate)

    file_ERR = dir_err + survey + '-' + fil + '-error.txt'
    if survey == 'ceers':
        if fil == 'f200w':
            bkgsig = 0.00828638007198104
            ERR = np.loadtxt(file_ERR)

        if fil == 'f150w':
            bkgsig = 0.009413915528320422
            ERR = np.loadtxt(file_ERR)

        if fil == 'f356w':
            bkgsig = 0.002497224970380189
            ERR = np.loadtxt(file_ERR)
        
        if fil == 'f444w':
            bkgsig = 0.0038675931576882113
            ERR = np.loadtxt(file_ERR)


    if survey == 'smac0723':
        if fil == 'f200w':
            bkgsig = 0.005344214206900176
            ERR = np.loadtxt(file_ERR)
             
        if fil == 'f150w':
            bkgsig = 0.0066008897972703
            ERR = np.loadtxt(file_ERR)

        if fil == 'f356w':
            bkgsig = 0.0021360208138916452
            ERR = np.loadtxt(file_ERR)

        if fil == 'f444w':
            bkgsig = 0.003075707146720149
            ERR = np.loadtxt(file_ERR)

    if survey == 'abell-nircam':
        if fil == 'f200w':
            bkgsig = 0.007353399665513717
            ERR = np.loadtxt(file_ERR)

        if fil == 'f150w':
            bkgsig = 0.00822649445522193
            ERR = np.loadtxt(file_ERR)
        
        if fil == 'f356w':
            bkgsig = 0.0020468222088078983
            ERR = np.loadtxt(file_ERR)

        if fil == 'f444w':
            bkgsig = 0.0015385094554139184
            ERR = np.loadtxt(file_ERR)


    if survey == 'abell-niriss':
        if fil == 'f200w':
            bkgsig = 0.00394144303781326
            ERR = np.loadtxt(file_ERR)

        if fil == 'f150w':
            bkgsig = 0.004077401012265584
            ERR = np.loadtxt(file_ERR)

    return bkgsig, rate, ERR

def convert_MJySr_to_10nJy(img, pixscl):
    pixarea = pixscl**2*(4.848137E-6)**2
    return img * 1e14 * pixarea

def convert_10nJy_to_MJySr(img, pixscl):
    pixarea = pixscl**2*(4.848137E-6)**2
    return img / 1e14 / pixarea


    
def positive_pa(pa):
    while pa < 0:
        pa += 180
    while pa > 180:
        pa -= 180
    return pa


def get_initial_params(img, eps, PA):
#     print(img.shape[0])    
    DIM = img.shape[0]
    xc = DIM/2
    yc = DIM/2
    iso = extract_fix_isophotes(image=img, xcen=xc, ycen=yc, eps=eps, pa=PA, step=0.1, initsma = 3, silent=True)
    xarr = iso.tflux_e/np.max(iso.tflux_e)
    yarr = iso.sma
    f = interpolate.interp1d(xarr, yarr, kind='linear')
    r_e = f(0.5)
    f = interpolate.interp1d(iso.sma, iso.intens, kind='linear')
    I_e = f(r_e).item()
    
    return r_e, I_e


def sersicfit(img, sigmap, psf, PA, eps, n, I_e, r_e):
    # create a imfit class. Let call this model `model_desc`
    model_desc = pyimfit.SimpleModelDescription()

    DIM = img.shape[0]

    model_desc.x0.setValue(DIM/2)
    model_desc.y0.setValue(DIM/2)

    # Creates an Single Sersic component
    sersic = pyimfit.make_imfit_function('Sersic', label='global')
    # Set initial values, lower and upper limits of each parameters
    # You can pretty much eyeball the PA of the disk or use the parameters derived from sep.
    sersic.PA.setValue(PA)
    sersic.ell.setValue(eps, [0, 1])
    sersic.n.setValue(2, [0.5, 10])
    sersic.I_e.setValue(I_e)
    sersic.r_e.setValue(r_e)

    model_desc.addFunction(sersic)
    # print(model_desc)

    # We want to pass the PSF image to the model object now
    if psf is None:
        imfitter = pyimfit.Imfit(model_desc)
    else:
        imfitter = pyimfit.Imfit(model_desc, psf)

    imfitter.loadData(img, error=sigmap, error_type="sigma")

    result = imfitter.doFit(solver='LM')

    output = {}

    output['fitConverged'] = int(result.fitConverged)
    output['reducedCHI2'] = round(result.fitStatReduced, 5)
    for jj in range(len(imfitter.numberedParameterNames)):
        name = imfitter.numberedParameterNames[jj][0:-2]
        output[name] = round(result.params[jj], 5) if result.fitConverged == True else -999
        output[name+'_err'] = round(result.paramErrs[jj], 5) if result.fitConverged == True else -999
    
    if result.fitConverged == True:
        bestfit_model_im = imfitter.getModelImage()
        (totalFlux, componentFluxes) = imfitter.getModelFluxes()
        residual_im = img - bestfit_model_im
    
        output['totalFlux'] = totalFlux
        output['bestfit'] = bestfit_model_im
        output['residual'] = residual_im
    else:
        output['totalFlux'] = -999
        output['bestfit'] = -999
        output['residual'] = -999

    return output

def over_sersicfit(img, sigmap, psf, PA, eps, n, I_e, r_e, scale):
    # create a imfit class. Let call this model `model_desc`
    model_desc = pyimfit.SimpleModelDescription()

    DIM = img.shape[0]

    model_desc.x0.setValue(DIM/2)
    model_desc.y0.setValue(DIM/2)

    # Creates an Single Sersic component
    sersic = pyimfit.make_imfit_function('Sersic', label='global')
    # Set initial values, lower and upper limits of each parameters
    # You can pretty much eyeball the PA of the disk or use the parameters derived from sep.
    sersic.PA.setValue(PA)
    sersic.ell.setValue(eps)
    sersic.n.setValue(2, [0.5, 10])
    sersic.I_e.setValue(I_e)
    sersic.r_e.setValue(r_e)

    model_desc.addFunction(sersic)
    # print(model_desc)

    # We want to pass the PSF image to the model object now
    imfitter = pyimfit.Imfit(model_desc)

    PsfOversampler=pyimfit.MakePsfOversampler(psf, scale, [0,DIM-1,0,DIM-1])
    imfitter.loadData(img, error=sigmap, error_type="sigma", psf_oversampling_list=[PsfOversampler])

    result = imfitter.doFit(solver='LM')

    output = {}

    output['fitConverged'] = int(result.fitConverged)
    output['reducedCHI2'] = round(result.fitStatReduced, 5)
    for jj in range(len(imfitter.numberedParameterNames)):
        name = imfitter.numberedParameterNames[jj][0:-2]
        output[name] = round(result.params[jj], 5) if result.fitConverged == True else -999
        output[name+'_err'] = round(result.paramErrs[jj], 5) if result.fitConverged == True else -999
    
    if result.fitConverged == True:
        bestfit_model_im = imfitter.getModelImage()
        (totalFlux, componentFluxes) = imfitter.getModelFluxes()
        residual_im = img - bestfit_model_im
    
        output['totalFlux'] = totalFlux
        output['bestfit'] = bestfit_model_im
        output['residual'] = residual_im
    else:
        output['totalFlux'] = -999
        output['bestfit'] = -999
        output['residual'] = -999

    return output



def run_statmorph(img, sigmap, psf, bkgsig=None):
    if bkgsig is None:
        background = Background_Mode(img)
        bkgsig = background['background noise']

    threshold = np.zeros_like(img)
    threshold[:] = bkgsig * 1.5

    npixels = 5 

    segm = photutils.detect_sources(img, threshold, npixels)
        # print(segm.slices)
    label = np.argmax(segm.areas) + 1 
    segmap = segm.data == label

    size = int( np.mean(img.shape)  * 0.1)
    segmap_float = ndi.uniform_filter(np.float64(segmap), size=size)
    segmap = segmap_float > 0.5 
    segmap = np.array(segmap, dtype=int)
    fits.writeto('test.fits', segmap, overwrite=True)

    start = time.time()
    source_morphs = statmorph.source_morphology(img, segmap, weightmap=sigmap, psf=psf, 
                        sky_mean=0, sky_median=0, sky_sigma=bkgsig, n_sigma_outlier=1000)
    print('Statmorph time: %g s.' % (time.time() - start))
    morph = source_morphs[0]
    
    return morph


def feed_statmorph(name, morph_dict, morph, suffix='', fact=1):
    morph_dict[name]['xc_centroid'+suffix] = morph.xc_centroid * fact
    morph_dict[name]['yc_centroid'+suffix] = morph.yc_centroid * fact
    morph_dict[name]['ellipticity_centroid'+suffix] = morph.ellipticity_centroid
    morph_dict[name]['orientation_centroid'+suffix] = morph.orientation_centroid
    morph_dict[name]['xc_asymmetry'+suffix] = morph.xc_asymmetry * fact
    morph_dict[name]['yc_asymmetry'+suffix] = morph.yc_asymmetry * fact
    morph_dict[name]['ellipticity_asymmetry'+suffix] = morph.ellipticity_asymmetry
    morph_dict[name]['orientation_asymmetry'+suffix] = morph.orientation_asymmetry
    morph_dict[name]['rpetro_circ'+suffix] = morph.rpetro_circ * fact
    morph_dict[name]['rpetro_ellip'+suffix] = morph.rpetro_ellip * fact
    morph_dict[name]['rhalf_circ'+suffix] = morph.rhalf_circ * fact
    morph_dict[name]['rhalf_ellip'+suffix] = morph.rhalf_ellip * fact
    morph_dict[name]['rmax_circ'+suffix] = morph.rmax_circ * fact
    morph_dict[name]['rmax_ellip'+suffix] = morph.rmax_ellip * fact

    morph_dict[name]['Gini'+suffix] = morph.gini
    morph_dict[name]['M20'+suffix] = morph.m20
    morph_dict[name]['F_GM'+suffix] = morph.gini_m20_bulge
    morph_dict[name]['S_GM'+suffix] = morph.gini_m20_merger 
    morph_dict[name]['C'+suffix] = morph.concentration
    morph_dict[name]['A'+suffix] = morph.asymmetry
    morph_dict[name]['S'+suffix] = morph.smoothness
    morph_dict[name]['M'+suffix] = morph.multimode
    morph_dict[name]['I'+suffix] = morph.intensity
    morph_dict[name]['D'+suffix] = morph.deviation
    morph_dict[name]['OA'+suffix] = morph.outer_asymmetry
    morph_dict[name]['SA'+suffix] = morph.shape_asymmetry

    morph_dict[name]['r20'+suffix] = morph.r20 * fact
    morph_dict[name]['r50'+suffix] = morph.r50 * fact
    morph_dict[name]['r80'+suffix] = morph.r80 * fact
    morph_dict[name]['flag'+suffix] = morph.flag

    morph_dict[name]['sersic_amp'+suffix] = morph.sersic_amplitude
    morph_dict[name]['sersic_rhalf'+suffix] = morph.sersic_rhalf
    morph_dict[name]['sersic_n'+suffix] = morph.sersic_n
    return morph_dict

def feed_imfit(name, morph_dict, output, suffix=''):
    morph_dict[name]['fitConverged'+suffix] = output['fitConverged']
    morph_dict[name]['reducedCHI2'+suffix] = output['reducedCHI2']
    morph_dict[name]['X0'+suffix] = output['X0']
    morph_dict[name]['X0_err'+suffix] = output['X0_err']
    morph_dict[name]['Y0'+suffix] = output['Y0']
    morph_dict[name]['Y0_err'+suffix] = output['Y0_err']
    morph_dict[name]['PA'+suffix] = output['PA']
    morph_dict[name]['PA_err'+suffix] = output['PA_err']
    morph_dict[name]['ell'+suffix] = output['ell']
    morph_dict[name]['ell_err'+suffix] = output['ell_err']
    morph_dict[name]['n'+suffix] = output['n']
    morph_dict[name]['n_err'+suffix] = output['n_err']
    morph_dict[name]['I_e'+suffix] = output['I_e']
    morph_dict[name]['I_e_err'+suffix] = output['I_e_err']
    morph_dict[name]['r_e'+suffix] = output['r_e']
    morph_dict[name]['r_e_err'+suffix] = output['r_e_err']
    morph_dict[name]['totalFlux'+suffix] = output['totalFlux']
    
    return morph_dict

def feed_galfit(name, morph_dict, output, suffix=''):
    morph_dict[name]['reducedCHI2'+suffix] = output['CHI2NU']
    morph_dict[name]['X0'+suffix] = output['1_XC']
    morph_dict[name]['X0_err'+suffix] = output['1_XC_err']
    morph_dict[name]['Y0'+suffix] = output['1_YC']
    morph_dict[name]['Y0_err'+suffix] = output['1_YC_err']
    morph_dict[name]['Mag'+suffix] = output['1_MAG']
    morph_dict[name]['Mag_err'+suffix] = output['1_MAG_err']
    morph_dict[name]['r_e'+suffix] = output['1_RE']
    morph_dict[name]['r_e_err'+suffix] = output['1_RE_err']
    morph_dict[name]['n'+suffix] = output['1_N']
    morph_dict[name]['n_err'+suffix] = output['1_N_err']
    morph_dict[name]['AR'+suffix] = output['1_AR']
    morph_dict[name]['AR_err'+suffix] = output['1_AR_err']
    morph_dict[name]['PA'+suffix] = output['1_PA']
    morph_dict[name]['PA_err'+suffix] = output['1_PA_err']
    
    return morph_dict









def feed_cas(name, cas_dict, output, suffix=''):
    cas_dict[name]['xc_centroid'+suffix] = round(output['xc_centroid'], 2)
    cas_dict[name]['yc_centroid'+suffix] = round(output['yc_centroid'], 2)
    cas_dict[name]['xc_asym'+suffix] = round(output['xc_asymmetry'], 2)
    cas_dict[name]['yc_asym'+suffix] = round(output['yc_asymmetry'], 2)
    cas_dict[name]['eps_centroid'+suffix] = round(output['eps_centroid'], 3)
    cas_dict[name]['pa_centroid'+suffix] = round(output['pa_centroid'], 1)
    cas_dict[name]['rp_circ_centroid'+suffix] = round(output['rp_circ_centroid'], 2)
    cas_dict[name]['rp_ellp_centroid'+suffix] = round(output['rp_ellp_centroid'], 2)
#    cas_dict[name]['xc_asym_ref'+suffix] = round(output['xc_asymmetry_reflect'], 2)
#    cas_dict[name]['yc_asym_ref'+suffix] = round(output['yc_asymmetry_reflect'], 2)
    cas_dict[name]['good_area'+suffix] = round(output['good_area'], 2)
    cas_dict[name]['bad_area'+suffix] = round(output['bad_area'], 2)
    cas_dict[name]['tot_area'+suffix] = round(output['tot_area'], 2)
    cas_dict[name]['flux_ap'+suffix] = round(output['flux_ap'], 2)
    cas_dict[name]['r20'+suffix] = round(output['r20'], 2)
    cas_dict[name]['r50'+suffix] = round(output['r50'], 2)
    cas_dict[name]['r80'+suffix] = round(output['r80'], 2)
    cas_dict[name]['r90'+suffix] = round(output['r90'], 2)
    cas_dict[name]['A_cas'+suffix] = round(output['A_full'], 4)
    cas_dict[name]['A'+suffix] = round(output['A'], 4)

    cas_dict[name]['Ac'+suffix] = round(output['Acor'], 4)
    cas_dict[name]['f1'+suffix] = round(output['f1'], 4)
    cas_dict[name]['f2'+suffix] = round(output['f2'], 4)
    cas_dict[name]['delta1_pre'+suffix] = round(output['delta1_pre'], 4)
    cas_dict[name]['delta2_pre'+suffix] = round(output['delta2_pre'], 4)

#    cas_dict[name]['A_ref'+suffix] = round(output['A_r'], 4)
#    cas_dict[name]['A_cas_c'+suffix] = round(output['A_full_c'], 4)
#    cas_dict[name]['A_ref_c'+suffix] = round(output['A_r_c'], 4)

    cas_dict[name]['Asky_cas'+suffix] = round(output['Asky_cas'], 4)
    cas_dict[name]['Asky_min'+suffix] = round(output['Asky_min'], 4)
#    cas_dict[name]['Asky_ref'+suffix] = round(output['Asky_ref'], 4)
    cas_dict[name]['C'+suffix] = round(output['C'], 4)
#    cas_dict[name]['S'+suffix] = round(output['S'], 4)

    return cas_dict



def get_initial_params(img, eps, PA):
    frb = 1
    if img.shape[0] > 151:
        frb = img.shape[0]/151
        img = frebin(img, (151, 151), total=False)
    DIM = img.shape[0]
    xc = DIM/2
    yc = DIM/2
    iso = extract_fix_isophotes(image=img, xcen=xc, ycen=yc, eps=eps, pa=PA, step=0.1, initsma = 3, silent=True)
    xarr = iso.tflux_e/np.max(iso.tflux_e)
    yarr = iso.sma
    f = interpolate.interp1d(xarr, yarr, kind='linear')
    r_e = f(0.5)
    f = interpolate.interp1d(iso.sma, iso.intens, kind='linear')
    I_e = f(r_e).item()
    
    return r_e*frb, I_e


def get_transition_psf(fwhm, b, psf_high):

    psf_rb = get_psf_moffat_imfit(fwhm=fwhm, beta=b)
    psf_rb = make_same_DIM_with(psf_rb, psf_high.shape )
    psf_rb = psf_rb / np.sum(psf_rb)

    psf_tran = create_matching_kernel(psf_rb, psf_high)
    psf_tran[psf_tran<0.00001] = 0.00001

    psf_tran = psf_tran / np.sum(psf_tran)
    psf_rec = convolve(psf_rb, psf_tran, boundary='fill', fill_value=0)

    return psf_tran, psf_rec


def get_initial_epa(img):
    
    background = Background_Mode(img)
    bkgsig = background['background noise']

    threshold = np.zeros_like(img)
    threshold[:] = bkgsig * 1.5
    npixels = 5

    segm = photutils.detect_sources(img, threshold, npixels)

    label = np.argmax(segm.areas) + 1
    segmap = segm.data == label

    size = int( np.mean(img.shape)  * 0.1)
    segmap_float = ndi.uniform_filter(np.float64(segmap), size=size)
    segmap = segmap_float > 0.5
    segmap = np.array(segmap, dtype=int)

    segmap = photutils.SegmentationImage(segmap)
    
    mask_bkg = segmap == 0

    img_source = np.where(~mask_bkg, img, 0.0) # background regions are set to zero

    image = np.float64(img_source) 

    # Calculate centroid
    M = skimage.measure.moments(image, order=1)
    xc_centroid = M[1, 0] / M[0, 0]
    yc_centroid = M[0, 1] / M[0, 0]

    Mc = skimage.measure.moments_central(image, center=(yc_centroid, xc_centroid), order=2)
    assert Mc[0, 0] > 0

    covariance = np.array([
                          [Mc[0, 2], Mc[1, 1]],
                          [Mc[1, 1], Mc[2, 0]]])
    covariance /= Mc[0, 0]  # normalize

    rho = 1.0 / 12.0  # variance of 1 pixel-wide top-hat distribution
    x2, xy, xy, y2 = covariance.flat
    while np.abs(x2*y2 - xy**2) < rho**2:
        x2 += (x2 >= 0) * rho - (x2 < 0) * rho  # np.sign(0) == 0 is no good
        y2 += (y2 >= 0) * rho - (y2 < 0) * rho

    covariance_centroid = np.array([[x2, xy],
                                    [xy, y2]])
    eigvals = np.linalg.eigvals(covariance_centroid)
    eigvals_centroid = np.sort(np.abs(eigvals))[::-1]

    a = np.sqrt(np.abs(eigvals_centroid[0]))
    b = np.sqrt(np.abs(eigvals_centroid[1]))

    eps_centroid = 1.0 - (b / a)
    ############# The orientation (in radians) of the source

    x2, xy, xy, y2 = covariance_centroid.flat

    orientation_centroid = 0.5 * np.arctan2(2.0 * xy, x2 - y2)

    pa_centroid = positive_pa(-90 + orientation_centroid * 180. / np.pi)

    return xc_centroid, yc_centroid, eps_centroid, pa_centroid



def noisy(img_10nJy, pixscl=None, survey='ceers', bpass='f200w'):

    nonoise_10nJy = copy.deepcopy(img_10nJy)
    nonoise_10nJy[nonoise_10nJy<0] = 0 

    source_err = np.zeros_like(nonoise_10nJy)

    bkgsig, rate, ERR = get_jwst_noise_array(survey=survey, fil=bpass, pixscl=pixscl) # in nJy

    if np.max(nonoise_10nJy) > np.max(rate):
        raise ValueError('data too high value, check the noisy.py: '+str(np.max(nonoise_10nJy)))

    f = interpolate.interp1d(rate, ERR, kind='linear', fill_value="extrapolate")

    source_err[:] = f(nonoise_10nJy[:])
    map_sig_10nJy = np.sqrt(bkgsig**2 + source_err**2)     # in MJy/sr

    img_noisy_10nJy = img_10nJy + map_sig_10nJy * np.random.randn(img_10nJy.shape[0], img_10nJy.shape[1])
    tmp = copy.deepcopy(img_noisy_10nJy)
    tmp[tmp<0] = 0 

    source_err_noisy = f(tmp)
    map_sig_noisy_10nJy = np.sqrt(bkgsig**2 + source_err_noisy**2)

    return img_noisy_10nJy, map_sig_noisy_10nJy, map_sig_10nJy



def get_jwst_noise_array(survey='ceers', fil='f200w', pixscl=None):
    '''
    see which notebook? Please add pixscl as a argument: /Users/syu/works/MissSpirals/code/JWST
    '''
    path_jwst_noise = '/Users/syu/works/MissSpirals/code/JWST/'
    survey = survey.lower()
    fil = fil.lower()
    r = {'f150w':pixscl/0.031, 'f200w':pixscl/0.031, 'f356w':pixscl/0.063, 'f444w':pixscl/0.063}

    file_rate = path_jwst_noise + 'source_rate_in_nJy.txt'
    rate = np.loadtxt(file_rate) * r[fil]**2

    file_ERR = path_jwst_noise + survey + '-' + fil + '-error.txt'
    ERR = np.loadtxt(file_ERR) * r[fil]

    bkgsig_ceers = {'f150w':0.02126, 'f200w':0.01871, 'f356w':0.023296, 'f444w':0.036080}
    bkgsig_smac0723 = {'f150w':0.014909, 'f200w':0.01207, 'f356w':0.019926, 'f444w':0.028692}
    bkgsig_abell_nircam = {'f150w':0.01858, 'f200w':0.01661, 'f356w':0.019094, 'f444w':0.014352}
    bkgsig_abell_niriss = {'f150w':0.009209, 'f200w':0.008902}

    bkgsig_store = {'ceers':bkgsig_ceers, 'smac0723':bkgsig_smac0723, 'abell-nircam':bkgsig_abell_nircam, 'abell-niriss':bkgsig_abell_niriss}
    bkgsig = bkgsig_store[survey][fil] * r[fil]

    return bkgsig, rate, ERR


def write_params_galfit(path_img, path_imgblock, path_sig, path_psf, scale, DIM, m, r_e, eps, PA, outpath):
    lines = [ 
    'A) '+path_img+'            # Input data image (FITS file)',
    'B) '+path_imgblock+'       # Output data image block',
    'C) '+path_sig+'          # Sigma image name (made from data if blank or "none")',
    'D) '+path_psf+'   #        # Input PSF image and (optional) diffusion kernel',
    'E) '+str(scale)+'                   # PSF fine sampling factor relative to data ',
    'F) none                # Bad pixel mask (FITS image or ASCII coord list)',
    'G) none                # File with parameter constraints (ASCII file) ',
    'H) 1 '+str(DIM)+' 1 '+str(DIM)+'           # Image region to fit (xmin xmax ymin ymax)',
    'I) '+str(DIM)+' '+str(DIM)+'                   # Size of the convolution box (x y)',
    'J) 28.90               # Magnitude photometric zeropoint ',
    'K) 0.031  0.031        # Plate scale (dx dy)    [arcsec per pixel]',
    'O) regular             # Display type (regular, curses, both)',
    'P) 0                   # Choose: 0=optimize, 1=model, 2=imgblock, 3=subcomps',
    ' 0) sersic                 #  object type',
    ' 1) '+str(DIM/2)+'  '+str(DIM/2)+'   1 1  #  position x, y',
    ' 3) '+str(m)+'     1          #  Integrated magnitude   ',
    ' 4) '+str(r_e)+'     1          #  R_e (half-light radius)   [pix]',
    ' 5) 2.2490      1          #  Sersic index n (de Vaucouleurs n=4) ',
    ' 6) 0.0000      0          #     ----- ',
    ' 7) 0.0000      0          #     ----- ',
    ' 8) 0.0000      0          #     ----- ',
    ' 9) '+str(1-eps)+'      1          #  axis ratio (b/a)  ',
    '10) '+str(PA)+'    1          #  position angle (PA) [deg: Up=0, Left=90]',
    ' Z) 0                      #  output option (0 = resid., 1 = Dont subtract)']
    with open(outpath, 'w') as f:
        for line in lines:
            f.write(line)
            f.write('\n')




    return 


def galfit_sersicfit(img, sigmap, psf, DIM, r_e, eps, PA, scale=1):
    fits.writeto('tmp_img.fits', img, overwrite=True)
    fits.writeto('tmp_sig.fits', sigmap, overwrite=True)
    fits.writeto('tmp_psf.fits', psf, overwrite=True)
    m = -2.5*np.log10(np.sum(img)) + 28.90

    write_params_galfit('tmp_img.fits', 'tmp_imgblock.fits', 'tmp_sig.fits',
        'tmp_psf.fits', scale, DIM, m, r_e, eps, PA, 'input.params')

    bashCommand = '/Users/syu/Software/galfit3.0/galfit input.params'
    process = subprocess.run(bashCommand.split())

    HDU = fits.open('tmp_imgblock.fits')
    hdr_fit = HDU[2].header
    output = {}
    keys = ['1_XC', '1_YC', '1_MAG', '1_RE', '1_N', '1_AR', '1_PA']
    for key in keys:
        index = hdr_fit[key].find('+')
        output[key] = hdr_fit[key][0:index]
        output[key+"_err"] = hdr_fit[key][index+3:]

    output['CHI2NU'] = hdr_fit['CHI2NU']

    return output



def error_intensity2mu(intens=None, err_intens=None):
    if intens is None or err_intens is None:
        print('-- grammar: error_intensity2mu(intens=None, err_intens=None) --')
        return
    return 2.5/np.log(10)*err_intens/intens





import importlib
import os
from astropy.table import Table
import numpy as np
from astropy.table import Table, Column
from pathlib import Path
from os import path
import pandas as pd

from Taiji.imtools import removeellipseIndef
from Taiji.imtools import bright_to_mag
from Taiji.imtools import symmetry_propagate_err_mu
from Taiji.imtools import normalize_angle
from Taiji.imtools import correct_pa_profile

package_name = 'pyraf'

if importlib.util.find_spec(package_name) is not None:
    from pyraf import iraf

    iraf.stsdas()
    iraf.analysis()
    iraf.isophote()
    
else:
    print(f"Package {package_name} is not installed.")

def maskFitsTool(inputImg_fits_file, mask_fits_file):

    # firstly, I should judge the file is fits or fit.
    if inputImg_fits_file[-3:] == 'its':
        mask_pl_file = inputImg_fits_file.replace('.fits', '.fits.pl')
    elif inputImg_fits_file[-3:] == 'fit':
        mask_pl_file = inputImg_fits_file.replace('.fit', '.pl')

    if os.path.exists(mask_pl_file):
        print('pl file exists')
    else:
        print(
            'pl file does not exist and we should make a pl file for ellipse task'
        )
        iraf.imcopy(mask_fits_file, mask_pl_file)

def PyrafEllipse(input_img,
                 outTab,
                 outDat,
                 cdf,
                 pf,
                 inisma,
                 maxsma,
                 x0,
                 y0,
                 pa,
                 ell_e,
                 zpt0,
                 interactive=False,
                 inellip='',
                 hcenter=False,
                 hpa=False,
                 hellip=False,
                 nclip=3,
                 usclip=3,
                 lsclip=2.5,
                 FracBad=0.9,
                 olthresh=0,
                 intemode='median',
                 step=0.1,
                 sky_err=0,
                 sky_value=0,
                 maxgerr=0.5,
                 harmonics=False,
                 texp=1,
                 pixel_size=0.259):

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

    iraf.ellipse(interactive=interactive,
                 input=input_img,
                 out=outTab,
                 fflag=FracBad,
                 sma0=inisma,
                 maxsma=maxsma,
                 x0=x0,
                 nclip=nclip,
                 usclip=usclip,
                 lsclip=lsclip,
                 y0=y0,
                 pa=pa,
                 e=ell_e,
                 hcenter=hcenter,
                 hpa=hpa,
                 hellip=hellip,
                 olthresh=olthresh,
                 integrmode=intemode,
                 step=step,
                 inellip=inellip,
                 maxgerr=maxgerr,
                 harmonics=harmonics)

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
    dPA = 75
    ellipse_data['pa'] = removeellipseIndef(ellipse_data['pa'])
    ellipse_data = correct_pa_profile(ellipse_data, delta_pa=dPA)
    ellipse_data.add_column(
        Column(name='pa_norm', data=np.array(
            [normalize_angle(pa, lower=0, upper=180.0, b=False) 
             for pa in ellipse_data['pa']])))

    # remove the indef
    intens = ellipse_data['intens']
    intens_err = ellipse_data['int_err']
    intens_err_removeindef = removeellipseIndef(intens_err)

    ellipse_data['ell'] = removeellipseIndef(ellipse_data['ell'])
    ellipse_data['pa'] = removeellipseIndef(ellipse_data['pa'])
    ellipse_data['ell_err'] = removeellipseIndef(ellipse_data['ell_err'])
    ellipse_data['pa_err'] = removeellipseIndef(ellipse_data['pa_err'])
    
    ellipse_data['x0'] = removeellipseIndef(ellipse_data['x0'])
    ellipse_data['x0_err'] = removeellipseIndef(ellipse_data['x0_err'])
    ellipse_data['y0'] = removeellipseIndef(ellipse_data['y0'])
    ellipse_data['y0_err'] = removeellipseIndef(ellipse_data['y0_err'])

    # calculate the magnitude.
    intens_err_removeindef_sky = np.sqrt(
        np.array(intens_err_removeindef)**2 + sky_err**2)
    
    if sky_value:
        ellipse_data['intens'] = np.array(intens) - sky_value
        mu = bright_to_mag(intens - sky_value, zpt0, texp, pixel_size)
        mu_err = symmetry_propagate_err_mu(
        np.array(intens) - sky_value, intens_err_removeindef_sky)
    else:
        mu = bright_to_mag(intens, zpt0, texp, pixel_size)
        mu_err = symmetry_propagate_err_mu(
        np.array(intens), intens_err_removeindef_sky)
    
    ellipse_data.add_column(Column(name='mu', data=mu))
    ellipse_data.add_column(Column(name='mu_err', data=mu_err))

    return ellipse_data

def extract_autoprof_aux(aux_path_arr):

    ell_arr = np.full((len(aux_path_arr)), np.nan)
    err_ell_arr = np.full((len(aux_path_arr)), np.nan)
    pa_arr = np.full((len(aux_path_arr)), np.nan)
    err_pa_arr = np.full((len(aux_path_arr)), np.nan)
    rad_pix_arr = np.full((len(aux_path_arr)), np.nan)

    numpass_arr = np.full((len(aux_path_arr)), np.nan)
    check_fit_fft_coefficients = np.full((len(aux_path_arr)), np.nan)
    check_fit_light_symmetry = np.full((len(aux_path_arr)), np.nan)
    check_fit_initial_fit_compare = np.full((len(aux_path_arr)), np.nan)
    check_fit_isophote_variability = np.full((len(aux_path_arr)), np.nan)

    for ind, aux_path in enumerate(aux_path_arr):
        if path.exists(aux_path):
            with open(aux_path, "r") as file_aux:
                num_pass = 0
                for line in file_aux:
                    if line.startswith("checkfit"):
                        num_pass = num_pass + 1 if line.strip(
                        )[-4:] == "pass" else num_pass

                        numpass_arr[ind] = num_pass

                    if line.startswith("checkfit FFT"):
                        check_fit_fft_coefficients[ind] = 1 if line.strip(
                        )[-4:] == "pass" else 0
                    if line.startswith("checkfit Light"):
                        check_fit_light_symmetry[ind] = 1 if line.strip(
                        )[-4:] == "pass" else 0
                    if line.startswith("checkfit initial"):
                        check_fit_initial_fit_compare[ind] = 1 if line.strip(
                        )[-4:] == "pass" else 0
                    if line.startswith("checkfit isophote"):
                        check_fit_isophote_variability[ind] = 1 if line.strip(
                        )[-4:] == "pass" else 0

                    hua_mod = 1
                    if line.startswith("global optimal ellipticity") and num_pass >= 0:
                        ell, err_ell, pa, err_pa, rad_pix = np.array(
                            line.split())[[2+hua_mod, 4+hua_mod, 6+hua_mod, 8+hua_mod, 11+hua_mod]]
                        if float(rad_pix
                                ) > 0:  # rad_pix should be larger than 3*seeing
                            ell_arr[ind] = float(ell)
                            if err_ell[-1] == ",":
                                err_ell_arr[ind] = float(err_ell[:-1])
                            else:
                                print("error!")
                                err_ell_arr[ind] = np.nan
                            pa_arr[ind] = float(pa)
                            err_pa_arr[ind] = float(err_pa)
                            rad_pix_arr[ind] = float(rad_pix)

    # return a pd dataframe
    return pd.DataFrame({
        "ell": ell_arr,
        "err_ell": err_ell_arr,
        "pa": pa_arr,
        "err_pa": err_pa_arr,
        "rad_pix": rad_pix_arr,
        "numpass": numpass_arr,
        "check_fit_fft_coefficients": check_fit_fft_coefficients,
        "check_fit_light_symmetry": check_fit_light_symmetry,
        "check_fit_initial_fit_compare": check_fit_initial_fit_compare,
        "check_fit_isophote_variability": check_fit_isophote_variability
    })
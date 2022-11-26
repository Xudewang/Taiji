import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
from astropy.io import fits


def make_galfit_fitarea(image_file, fitarea=None):
    if fitarea is None:
        image_data = fits.getdata(image_file)
        ylen, xlen = image_data.shape
        fitarea = np.array(((0, xlen - 1), (0, ylen - 1)))

    fitarea = np.array(fitarea) + 1
    return fitarea


def make_head(runoption_num,
              output_file,
              image_file,
              psf_file,
              mask_file,
              sigma_file,
              convbox,
              mag_zpt0,
              pixel_scale,
              constraints_file=None,
              fitarea=None):

    fitarea = make_galfit_fitarea(image_file, fitarea)
    head = '# IMAGE and GALFIT CONTROL PARAMETERS'
    head += '\nA) {}'.format(image_file)
    head += '\nB) {}'.format(output_file)
    head += '\nC) {}'.format(sigma_file if sigma_file is not None else 'none')
    head += '\nD) {}'.format(psf_file if psf_file is not None else 'none')
    head += '\nE) {}'.format(1)
    head += '\nF) {}'.format('none' if mask_file is None else mask_file)
    head += '\nG) {}'.format(
        'none' if constraints_file is None else constraints_file + ' ')
    head += '\nH) {}   {}   {}   {}'.format(*fitarea[0], *fitarea[1])
    head += '\nI) {}  {}'.format(convbox, convbox)
    head += '\nJ) {}'.format(mag_zpt0)
    head += '\nK) {}  {}'.format(pixel_scale, pixel_scale)
    head += '\nO) {}'.format('regular')
    head += '\nP) {}'.format(runoption_num)
    return head


def Make_Sersic_parameters(value_arr, fixed_num_arr=np.array([1, 1, 1, 1, 1])):

    parameternumber_arr = [3, 4, 5, 9, 10]

    description_arr = [
        'Surface brightness at R_e [mag/arcsec^2]',
        'R_e (effective radius)   [pix]',
        'Sersic index n (de Vaucouleurs n=4)', 'Axis ratio (b/a)',
        'Position angle (PA) [deg: Up=0, Left=90]'
    ]

    entry_arr = [
        '\n{:2d}) {:8.4f}         {:d}        # {}'.format(
            parameternumber_arr[i], value_arr[i], fixed_num_arr[i],
            description_arr[i]) for i in range(len(parameternumber_arr))
    ]

    return ''.join(entry_arr)


def Make_Expdisk_parameters(value_arr, fixed_num_arr=np.array([1, 1, 1, 1])):

    parameternumber_arr = [3, 4, 9, 10]

    description_arr = [
        'Central surface brghtness [mag/arcsec^2]',
        'R_s (disk scale-length)   [pix]', 'Axis ratio (b/a)',
        'Position angle (PA) [deg: Up=0, Left=90]'
    ]

    entry_arr = [
        '\n{:2d}) {:8.4f}         {:d}        # {}'.format(
            parameternumber_arr[i], value_arr[i], fixed_num_arr[i],
            description_arr[i]) for i in range(len(parameternumber_arr))
    ]

    return ''.join(entry_arr)


def Make_Sky_parameters(value_arr, fixed_num_arr=np.array([1, 0, 0])):

    parameternumber_arr = [1, 2, 3]

    description_arr = ('bkg value at center of fitting region [ADUs]',
                       'dbkg / dx (bkg gradient in x)',
                       'dbkg / dy (bkg gradient in y)')

    entry_arr = [
        '\n{:2d}) {:8.4f}         {:d}        # {}'.format(
            parameternumber_arr[i], value_arr[i], fixed_num_arr[i],
            description_arr[i]) for i in range(len(parameternumber_arr))
    ]

    return ''.join(entry_arr)


def Component_to_galfit(componentnumber,
                        name_arr,
                        parameter_arr,
                        fixed_arr,
                        skipinimage=False):
    body_arr = []
    for i in range(componentnumber):

        if name_arr[i] == "expdisk1":
            body = '# Component number: {}'.format(i + 1)
            body += '\n 0) {:25s} # Component type'.format(name_arr[i])
            body += '\n 1) {:8.4f}  {:8.4f}  {:d}  {:d}  # position x, y'.format(
                parameter_arr['cen_x'], parameter_arr['cen_y'],
                fixed_arr['cenx_fixed'], fixed_arr['ceny_fixed'])

            entry = Make_Expdisk_parameters(
                value_arr=[
                    parameter_arr['mu0'], parameter_arr['rs'],
                    parameter_arr['axisratio_disk'], parameter_arr['pa_disk']
                ],
                fixed_num_arr=[
                    fixed_arr['mu0_fixed'], fixed_arr['rs_fixed'],
                    fixed_arr['axisratio_disk_fixed'],
                    fixed_arr['pa_disk_fixed']
                ])
            body += entry

        if name_arr[i] == 'sersic2':
            body = '# Component number: {}'.format(i + 1)
            body += '\n 0) {:25s} # Component type'.format(name_arr[i])
            body += '\n 1) {:8.4f}  {:8.4f}  {:d}  {:d}  # position x, y'.format(
                parameter_arr['cen_x'], parameter_arr['cen_y'],
                fixed_arr['cenx_fixed'], fixed_arr['ceny_fixed'])
            entry = Make_Sersic_parameters([
                parameter_arr['mue'], parameter_arr['re'], parameter_arr['n'],
                parameter_arr['axisratio_bulge'], parameter_arr['pa_bulge']
            ])
            body += entry

        if name_arr[i] == 'sky':
            body = '# Component number: {}'.format(i + 1)
            body += '\n 0) {:25s} # Component type'.format(name_arr[i])

            entry = Make_Sky_parameters([
                parameter_arr['bkg'], parameter_arr['dbkg_dx'],
                parameter_arr['dbkg_dy']
            ])
            body += entry

        body += '\n Z) {:d}                         # {}'.format(
            skipinimage, 'Skip this model in output image?(yes=1, no=0)')
        body_arr.append('\n\n' + body)
    body_arr = ''.join(body_arr)

    return body_arr


def Galfit_fit(feedme_file, run_type, feedme_dir, code_dir=None):

    os.chdir(feedme_dir)
    print('feedme dir is: ', os.getcwd())

    galfitcmd = 'galfit'

    cmd = [galfitcmd, run_type, feedme_file]

    popen = subprocess.Popen([f'galfit {run_type} {feedme_file}'], shell=True)

    return_code = popen.wait()

    if return_code == 0:
        print('Galfit ran!')

    else:
        print('Galfit does not run!')
        
    # if Path('galfit.01').exists():
    #     #Path('galfit.01').unlink()
    #     print('galfit.01 exsits')
    # else:
    #     print('no galfit.01, does not run???')
    os.chdir(code_dir)
    print('code dir is: ', os.getcwd())
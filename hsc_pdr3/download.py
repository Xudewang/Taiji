import subprocess
import os
import numpy as np
from astropy.table import Table, Column, vstack
from astropy import units as u

from .utils import r_phy_to_ang
from . import query

ANG_UNITS = ['arcsec', 'arcsecond', 'arcmin', 'arcminute', 'deg']
PHY_UNITS = ['pc', 'kpc', 'Mpc']

def hsc_cutout_tool(rerun_field, data_type_command, coor_ra, coor_dec, size_arcsec, data_path, code_path):
    
    cutout_download_name = '{rerun}_{type}_{ra}_{dec}_{filter}_' + '{:.2f}arcsec_cutout'.format(size_arcsec)
    
    if data_type_command == 'coadd/bg':
        Datatype = 'coadd+bg'
    elif data_type_command == 'coadd':
        Datatype = 'coadd'
        
    filters_hsc = ['HSC-G', 'HSC-R', 'HSC-I', 'HSC-Z', 'HSC-Y']
    
    for filter_band in filters_hsc:
        check_data_file_name = data_path + '{0}_{1}_{2}_{3}_{4}_{5:.2f}arcsec_cutout.fits'.format(rerun_field, Datatype, coor_ra, coor_dec, filter_band, size_arcsec)
        if os.path.exists(check_data_file_name):
            print('The {} band cutout exists.'.format(filter_band))
            
        else:
            print('The {} band cutout does not exist.'.format(filter_band))
     
    check_data_file_name_Iband = data_path + '{0}_{1}_{2}_{3}_{4}_{5:.2f}arcsec_cutout.fits'.format(rerun_field, Datatype, coor_ra, coor_dec, 'HSC-I', size_arcsec)
    if os.path.exists(check_data_file_name_Iband):
        print('At least, HSC-I band cutout exsists, we stop the download code.')
        
    else:
        
        os.chdir(data_path)

        process = subprocess.Popen(["python /home/dewang/Taiji/hsc_pdr3/downloadCutout/downloadCutout.py --rerun='{0}' --type='{1}' --mask=True --variance=True --ra={2} --dec={3} \
                                    --sw={4}arcsec --sh={5}arcsec --name='{6}' --user='dwxu'".format(rerun_field, data_type_command, coor_ra, coor_dec, size_arcsec, size_arcsec, \
                                                                                                     cutout_download_name)], shell=True)

        return_code = process.wait()

        if return_code == 0:
            print('The download cutout process is successful!')

        os.chdir(code_path)
        
def hsc_psf_tool(rerun_field, data_type_command, coor_ra, coor_dec, size_arcsec, data_path, code_path):
    
    psf_download_name = '{rerun}_{type}_{ra}_{dec}_{filter}_' + '{:.2f}arcsec_psf'.format(size_arcsec)
    
    if data_type_command == 'coadd/bg':
        Datatype = 'coadd'
    elif data_type_command == 'coadd':
        Datatype = 'coadd'
        
    filters_hsc = ['HSC-G', 'HSC-R', 'HSC-I', 'HSC-Z', 'HSC-Y']
    
    for filter_band in filters_hsc:
        check_psf_file_name = data_path + '{0}_{1}_{2}_{3}_{4}_{5:.2f}arcsec_psf.fits'.format(rerun_field, Datatype, coor_ra, coor_dec, filter_band, size_arcsec)
        if os.path.exists(check_psf_file_name):
            print('The {} band PSF exists.'.format(filter_band))
        else:
            print('The {} band PSF dose not exist.'.format(filter_band))
            
    check_psf_file_name_Iband = data_path + '{0}_{1}_{2}_{3}_{4}_{5:.2f}arcsec_psf.fits'.format(rerun_field, Datatype, coor_ra, coor_dec, 'HSC-I', size_arcsec)
    if os.path.exists(check_psf_file_name_Iband):
        print('At least HSC-I band PSF exists, we stop this download.'.format(filter_band))
        
    else:
        
        os.chdir(data_path)

        process_psf = subprocess.Popen(["python /home/dewang/Taiji/hsc_pdr3/downloadPsf/downloadPsf.py --rerun='{0}' --type='coadd' --ra={1} --dec={2} --name='{3}' \
                                --user='dwxu'".format(rerun_field, coor_ra, coor_dec, psf_download_name)], shell=True)

        return_code_psf = process_psf.wait()

        if return_code_psf == 0:
            print('The download PSF process is successful!')

        os.chdir(code_path)
        
def hsc_query_tool(sql_file, catalog_file, dr_type, data_path, code_path):
    os.chdir(data_path)

    process_sql = subprocess.Popen(["python /home/dewang/Taiji/hsc_pdr3/hscReleaseQuery/hscReleaseQuery.py --user='dwxu' --release-version={} --nomail --skip-syntax-check \
                                    {} --format fits > {}".format(dr_type, sql_file, catalog_file)], shell=True)

    return_code_sql = process_sql.wait()
    
    if return_code_sql==0:
        print('The query is successful!')

    os.chdir(code_path)
    
def _get_cutout_size(cutout_size, redshift=None, cosmo=None, verbose=True):
    """Parse the input for the size of the cutout."""
    if not isinstance(cutout_size, u.quantity.Quantity):
        if verbose:
            print("# Assume the cutout size is in arcsec unit.")
        cutout_size = cutout_size * u.Unit('arcsec')
        ang_size = cutout_size
    else:
        cutout_unit = cutout_size.unit
        if str(cutout_unit) in ANG_UNITS:
            ang_size = cutout_size.to(u.Unit('arcsec'))
        elif str(cutout_unit) in PHY_UNITS:
            if redshift is None:
                raise ValueError("# Need to provide redshift value to use physical size!")
            elif (redshift < 0.) or (~np.isfinite(redshift)):
                raise ValueError("# Redshift value is not valid!")
            else:
                ang_size = r_phy_to_ang(cutout_size, redshift, cosmo=cosmo)
        else:
            raise ValueError("# Wrong unit for cutout size: {}".format(str(cutout_unit)))

    return ang_size

def hsc_box_search(coord, box_size=10.0 * u.Unit('arcsec'), coord_2=None, redshift=None, archive=None, dr='pdr3', rerun='pdr3_wide', \
                   data_path='', code_path='', cosmo=None, verbose=True, **kwargs):
    """
    Search for objects within a box area.
    """

    # We use central coordinate and half image size as the default format.
    if coord_2 is None:
        if isinstance(box_size, list):
            if len(box_size) != 2:
                raise Exception("# Cutout size should be like: [Width, Height]")
            ang_size_w = _get_cutout_size(
                box_size[0], redshift=redshift, cosmo=cosmo, verbose=verbose)
            ang_size_h = _get_cutout_size(
                box_size[1], redshift=redshift, cosmo=cosmo, verbose=verbose)
        else:
            ang_size_w = ang_size_h = _get_cutout_size(
                box_size, redshift=redshift, cosmo=cosmo, verbose=verbose)
        ra_size = ang_size_w.to(u.Unit('deg'))
        dec_size = ang_size_h.to(u.Unit('deg'))
        ra1, ra2 = coord.ra.value - ra_size.value, coord.ra.value + ra_size.value
        dec1, dec2 = coord.dec.value - dec_size.value, coord.dec.value + dec_size.value
    else:
        ra1, dec1 = coord.ra.value, coord.dec.value
        ra2, dec2 = coord_2.ra.value, coord_2.dec.value
    
    sql_info = query.box_search(ra1, ra2, dec1, dec2, dr=dr, rerun=rerun, **kwargs)
    with open(os.path.join(data_path, "object.sql"), "w") as sql_file:
        sql_file.write("%s" % sql_info)
        
    hsc_query_tool(sql_file='object.sql', catalog_file='catalog.fits', dr_type=dr, data_path=data_path, code_path=code_path)
    
    objects = Table.read(os.path.join(data_path, 'catalog.fits'), format='fits')

    return objects
        
def hsc_cone_search(coord, radius=10.0 * u.Unit('arcsec'), redshift=None, dr='pdr2', rerun='pdr2_wide', cosmo=None,
                    verbose=True, data_path='', code_path='', **kwargs):
    """
    Search for objects within a cone area.
    """

    # We use central coordinate and half image size as the default format.
    ra, dec = coord.ra.value, coord.dec.value
    rad_arcsec = _get_cutout_size(
        radius, redshift=redshift, cosmo=cosmo, verbose=verbose).to(u.Unit('arcsec'))
    
    sql_info = query.cone_search(ra, dec, rad=rad_arcsec.value, dr=dr, rerun=rerun, **kwargs)

    with open(os.path.join(data_path, "object.sql"), "w") as sql_file:
        sql_file.write("%s" % sql_info)
        
    hsc_query_tool(sql_file='object.sql', catalog_file='catalog.fits', dr_type=dr, data_path=data_path, code_path=code_path)
    
    objects = Table.read(os.path.join(data_path, 'catalog.fits'), format='fits')

    return objects
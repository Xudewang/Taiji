import subprocess
import os

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
        
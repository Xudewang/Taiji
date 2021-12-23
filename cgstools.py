import sys 
sys.path.append('../')
import re
from astropy.io import fits
from astropy.io import ascii
import numpy as np
import os

from Taiji.imtools import readGalfitInput
from Taiji.imtools import subtract_sky
from Taiji.imtools import Remove_file


def getgalName(galaxy_name):
    galaxy_name = galaxy_name
    if galaxy_name[0]=='N':
        galnamesep2 = re.findall(r'[0-9]+|[A-Z]+', galaxy_name)
        if float(galnamesep2[1]) > 1000:
            galnamesep = galnamesep2
        elif float(galnamesep2[1]) < 1000:
            galnamesep = ['','']
            galnamesep[0]=galnamesep2[0]
            galnamesep[1]=str(int(float(galnamesep2[1])))
    elif galaxy_name[0] == 'I':
        galnamesep = re.findall(r'[0-9]+|[A-Z]+', galaxy_name)
    elif galaxy_name[0]=='E':
        galnamesep2 = re.findall(r'[0-9]+|[A-Z]+', galaxy_name)
        galnamesep = ['','']
        galnamesep[0]=galnamesep2[0]
        galnamesep[1] = galnamesep2[1]+'-G'+galnamesep2[2]
    
    galName = galnamesep[0]+' '+galnamesep[1]

    return np.array([galName, galnamesep])

def getinforHua(galaxy_name, file_path):
    """
    get the basis CGS information calculated by Hua.

    'file_path' is the path not including file name. Just tell where the file is. So this should be a string.
    """

    name_arr = getgalName(galaxy_name)
    #galName = name_arr[0] # NGC 3621
    galnamesep = name_arr[1] # [NGC, 3621]
    if galaxy_name == 'NGC4373A':
        galnamesep = ['NGC', '4373A']

    # table from Hua
    #TODO: in some codes, the file path is still not including '/'.
    coor_data = ascii.read('{0}/para_file/bul_cen.dat'.format(file_path), names=['name1','name2','x0','y0'])
    geo_data = ascii.read('{0}/para_file/bul_struct_param.bin'.format(file_path),names=['name1','name2','mtotal','mt_err','BT','BT_err',
                        'mue','mue_err','n','n_err','re','re_err','ellip','ellip_err','pa','pa_err','DT','bT','BL'])
    index_coor = (coor_data['name1']==galnamesep[0]) & (coor_data['name2']==galnamesep[1])
    index_geo = (geo_data['name1']==galnamesep[0]) & (geo_data['name2']==galnamesep[1])

    mue = geo_data[index_geo]['mue']

    Re = geo_data[index_geo]['re']/0.259 # unit is pixel

    n = geo_data[index_geo]['n']

    mue_err = geo_data[index_geo]['mue_err']
    Re_err = geo_data[index_geo]['re_err']/0.259
    n_err = geo_data[index_geo]['n_err']

    x0 = coor_data[index_coor]['x0']
    y0 = coor_data[index_coor]['y0']

    pa = geo_data[index_geo]['pa']

    ell = geo_data[index_geo]['ellip']
    
    BT = geo_data[index_geo]['BT']
    bT = geo_data[index_geo]['bT']
    DT = geo_data[index_geo]['DT']
    BT_err = geo_data[index_geo]['BT_err']
    
    #print(mue[0], mue_err[0], Re[0], Re_err[0], n[0], n_err[0], x0[0], y0[0], pa[0], ell[0])

    dict_info = {'mue':mue[0], 'mue_err':mue_err[0], 'Re':Re[0], 'Re_err':Re_err[0], 'n':n[0], 'n_err':n_err[0],\
     'x0':x0[0], 'y0':y0[0],'pa':pa[0], 'ell':ell[0], 'BT':BT[0], 'BT_err':BT_err[0], 'bT':bT[0]}

    return dict_info

 
def getGalfitSky(galaxy_name):

    # image path and data
    # Some images are fit, others are fits.
    imageFile_fit = '../disk_galaxies/{}/R/{}_R_reg.fit'.format(galaxy_name, galaxy_name)
    imageFile_fits = '../disk_galaxies/{}/R/{}_R_reg.fits'.format(galaxy_name, galaxy_name)
    image_clean_file = '../disk_galaxies/{}/R/{}_R_reg_clean.fits'.format(galaxy_name, galaxy_name)
    if os.path.exists(imageFile_fit):
        imageFile = imageFile_fit
    elif os.path.exists(imageFile_fits):
        imageFile = imageFile_fits
    image_header = fits.getheader(imageFile)
    image_clean_header = fits.getheader(image_clean_file)
        
    exp_time = image_header['old_expt']

    # To get sky value and its error from Hua's parametric file.
    dir_arr = ascii.read('../para_file/dir_GALFITmod', names=['path_gal','path_name', 'model_name'])  # path name etc.
    modelFile_arr = np.array(dir_arr['model_name'])
    gal_name_arr = np.array([re.findall('(?<=\/).*', dir_arr['path_gal'][i])[0] for i in range(len(dir_arr))])

    modelFile_name = modelFile_arr[gal_name_arr==galaxy_name][0]
    
    modelFile_path = '../para_file/galfit_output/'+galaxy_name+'_'+modelFile_name
    

    Galfit_input = readGalfitInput(modelFile_path)
    sky_value = float(Galfit_input[-1])/exp_time
    
    modelSkyupFile_name = modelFile_name.replace('.in','_up.in')
    modelSkyupFile_path = '../para_file/galfit_output/'+galaxy_name+'_'+modelSkyupFile_name
    try:
        if modelSkyupFile_name:

            Galfit_input_Skyup = readGalfitInput(modelSkyupFile_path)
            sky_value_Skyup = float(Galfit_input_Skyup[-1])/exp_time
            sky_err = sky_value_Skyup - sky_value

        elif image_header['sky_err']:
            sky_err = image_header['sky_err']

        elif image_clean_header['sky_err']:
            sky_err = image_clean_header['sky_err']

    except:
        print('You should calculate the sky error by yourself!!!')
        sky_err = np.nan

    return {'sky_value': sky_value, 'sky_err':sky_err}

def sky_HUA(galaxy_name, image_header):
    """
    get the sky value from Hua.
    
    input: galaxy name and header
    output: sky value and sky error
    
    """
    exp_time = image_header['old_expt']
    
    # To get sky value and its error from Hua's parametric file.
    dir_arr = ascii.read('../para_file/dir_GALFITmod', names=['path_gal','path_name', 'model_name'])  # path name etc.
    modelFile_arr = np.array(dir_arr['model_name'])
    gal_name_arr = np.array([re.findall('(?<=\/).*', dir_arr['path_gal'][i])[0] for i in range(len(dir_arr))])
    
    if galaxy_name in gal_name_arr:
        modelFile_name = modelFile_arr[gal_name_arr==galaxy_name][0]

        modelFile_path = '../para_file/galfit_output/'+galaxy_name+'_'+modelFile_name


        Galfit_input = readGalfitInput(modelFile_path)
        sky_value = float(Galfit_input[-1])/exp_time

        modelSkyupFile_name = modelFile_name.replace('.in','_up.in')
        modelSkyupFile_path = '../para_file/galfit_output/'+galaxy_name+'_'+modelSkyupFile_name
        
        if os.path.exists(modelSkyupFile_path):
            # because some galaxies sky values do not exist in Hua's parafile.

            Galfit_input_Skyup = readGalfitInput(modelSkyupFile_path)
            sky_value_Skyup = float(Galfit_input_Skyup[-1])/exp_time
            sky_err_galfit = sky_value_Skyup - sky_value
            sky_value_galfit = sky_value
        else:
            sky_err_galfit = np.nan
            sky_value_galfit = sky_value
        
    else:
        sky_value_galfit = np.nan
        sky_err_galfit = np.nan
    
    return np.array([sky_value_galfit, sky_err_galfit, galaxy_name])

def sky_ZY(image_header):
    """
    This function is to get the sky value/error and D25 from Zhaoyu products.
    input: image header list.
    output: sky value, sky error, D25 (arcsec)
    """
    
    if 'sky_val' in image_header:
        sky_value = image_header['sky_val']
        sky_err = image_header['sky_err']
        sky_type = 'direct method'
    elif 'sky' in image_header:
        sky_value = image_header['sky']
        sky_err = image_header['skyerr']
        sky_type = 'whatever method'
    else:
        sky_value = np.nan
        sky_err = np.nan
        sky_type = 'no sky'

    if 'D25' in image_header:
        D25 = image_header['D25']
    else:
        D25 = np.nan

    return np.array([sky_value, sky_err, D25, sky_type, image_header['object']])

def calculateCGSSky(galaxy_name, maxis = 1200, e_input = 0.5, pa_input = 40):
    galaxy_name = galaxy_name
    imageFile_fit = '../disk_galaxies/{}/R/{}_R_reg.fit'.format(galaxy_name, galaxy_name)
    imageFile_fits = '../disk_galaxies/{}/R/{}_R_reg.fits'.format(galaxy_name, galaxy_name)
    if os.path.exists(imageFile_fit):
        imageFile = imageFile_fit
    elif os.path.exists(imageFile_fits):
        imageFile = imageFile_fits
    data_fits_file = imageFile
    cleandata_fits_file = '../disk_galaxies/'+galaxy_name+'/R/'+galaxy_name+'_R_reg_clean.fits'
    mask_fits_file = '../disk_galaxies/'+galaxy_name+'/R/'+galaxy_name+'_R_reg_mm.fits'

    datahdu = fits.open(data_fits_file)
    parafile_from_header = fits.getheader(data_fits_file)
    clean_header = fits.getheader(cleandata_fits_file)
    maskhdu = fits.open(mask_fits_file)
    image = datahdu[0].data
    mask = maskhdu[0].data
        
    if 'cen_x' in parafile_from_header:
        x0 = parafile_from_header['cen_x']
        y0 = parafile_from_header['cen_y']
    elif 'cen_x' in clean_header:
        x0 = clean_header['cen_x']
        y0 = clean_header['cen_y']
    else:
        try:
            galfit_info = getinforHua(galaxy_name)
            x0 = galfit_info['x0']
            y0 = galfit_info['y0']
        except:
            print('no center coordinate can be found')
            
    if 'ell_e' in parafile_from_header:
        ellip = parafile_from_header['ell_e']
        PA = parafile_from_header['ell_pa']
    elif 'ell_e' in clean_header:
        ellip = clean_header['ell_e']
        PA = clean_header['ell_pa']
    else:
        try:
            galfit_info = getinforHua(galaxy_name)
            ellip = galfit_info['ell']
            PA = galfit_info['pa']
        except:
            print('no e and pa can be found')
            ellip = e_input
            PA = pa_input

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

def subtract_sky_cgs(input_file, mask_file, sky_value):
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
    
    #maskFitsTool(modfile, mask_file)                   

def getzpt0(fits_header):
    phot = fits_header['phot']
    if phot == 'Y':
        zpt0 = fits_header['zpt_lan']
    else:
        zpt0 = fits_header['zpt_gsc']
    
    return zpt0

# test part
if __name__ == '__main__':
   q = getgalName('IC4991')
   print(q, q[1][0])

   p = getinforHua('IC4991')
   print(p)
   print(p['x0'])

   w = getGalfitSky('IC4991')
   print(w)
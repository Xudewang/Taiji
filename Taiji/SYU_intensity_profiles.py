import matplotlib.pyplot as plt
from astropy.io import fits
import numpy as np
import os
from astropy.io import ascii
from astropy.table import Table, join
from astropy.nddata import Cutout2D
import subprocess

import numpy.ma as ma
from scipy.interpolate import griddata
from scipy import ndimage
import math


def list_my_functions():
    print("rho, phi = cart2pol(x, y) \nx, y = pol2cart(rho, phi) \ndict = dist_ellipse() \
          \nimg_depj = deproject() \nimg_proj = project() \ndict = Extract_azimuthal_linearStep() \
          \ndict = Extract_radial()")
    return

# list_my_functions()


def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return(rho, phi)

def pol2cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return(x, y)

def reducePA(pa):
    for jj in range(100):
        if pa > 180:
            pa -= 180
        if pa < 0:
            pa += 180
    return pa

def deproject(img=None, eps=None, pa=None, xcen=None, ycen=None):
    
    if img is None:
        print("deproject(img=, eps=, pa=, xcen=, ycen=)")
        return
    
    img_ang = ndimage.rotate(img, pa, reshape=False)

    nx = img.shape[1]
    ny = img.shape[0]

#     nx_mat = np.outer(np.ones(nx), np.arange(0, nx, 1))
#     ny_mat = np.outer(np.arange(0, ny, 1), np.ones(ny))
    nx_mat, ny_mat = np.meshgrid(np.arange(0, nx, 1), np.arange(0, ny, 1))    
    
    x_coord = nx_mat - xcen
    y_coord = ny_mat - ycen

    x_ba = x_coord * (1-eps) + xcen
    y_ba = y_coord + ycen
    img_depj = ndimage.map_coordinates(img_ang, [y_ba, x_ba])
    
    return img_depj


def project(img=None, eps=None, pa=None, xcen=None, ycen=None):

    if img is None:
        print("project(img=, eps=, pa=, xcen=, ycen=)")
        return
    
    nx = img.shape[1]
    ny = img.shape[0]

#     nx_mat = np.outer(np.ones(nx), np.arange(0, nx, 1))
#     ny_mat = np.outer(np.arange(0, ny, 1), np.ones(ny))
    nx_mat, ny_mat = np.meshgrid(np.arange(0, nx, 1), np.arange(0, ny, 1))
    
    x_coord = nx_mat - xcen
    y_coord = ny_mat - ycen

    x_ba = x_coord / (1-eps) + xcen
    y_ba = y_coord + ycen
    img_shrink = ndimage.map_coordinates(img, [y_ba, x_ba])
    
    img_proj = ndimage.rotate(img_shrink, -1*pa, reshape=False)

    return img_proj


def dist_ellipse(shape=None, nx=None, ny=None, xc=None, yc=None, ab=None , eps=None, 
                 PA=None, pa=None, xcen=None, ycen=None):
    
    ab = (1-eps) if eps is not None else ab
    PA = PA if PA is not None else pa
    xc = xcen if xcen is not None else xc
    yc = ycen if ycen is not None else yc
    
    # if shape is not None:
    #     print("shape = (ny, nx) NOT (nx, ny)")
    nx = shape[1] if shape is not None else nx
    ny = shape[0] if shape is not None else ny
    
    if nx is None:
        print("dist_ellipse(shape=, xcen=, ycen=, eps=, PA=, || nx=, ny=, xc=, yc=, ab=, pa=)")
        return
    
    PA = reducePA(PA)
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

    angle -= 90
    angle[angle<0]+=360
########
    nx_mat, ny_mat = np.meshgrid(np.arange(0, nx, 1), np.arange(0, ny, 1))    
    
    x_coord = nx_mat - xc
    y_coord = ny_mat - yc
    rho, angle_circular = cart2pol(x_coord, y_coord)
    angle_circular = angle_circular * 180./np.pi
    angle_circular -= 90
    angle_circular[angle_circular<0]+=360
    
    return {'angle':angle, 'sma':sma, 'angle_circular':angle_circular}


def Extract_azimuthal_linearStep(img=None, mask=None, width=None, Rbegin=None, Rend=None, step=None, 
                                        xcen=None, ycen=None, eps=None, PA=None):
    if img is None:
        print("Extract_azimuthal_linearStep(img=None, mask=None, " + \
              "width=None, Rbegin=None, Rend=None, step=None,xcen=None, ycen=None, eps=None, PA=None)")
        return

    if 'fit' in img:
        img = fits.open(img)[0].data
    if (mask is not None) and ('fit' in mask):
        mask = fits.open(mask)[0].data
    
    mask = np.zeros_like(img) if mask is None else mask
        
    dictionary = dist_ellipse(shape=img.shape, xcen=xcen, ycen=ycen, eps=eps, PA=PA)
    sma_mat = dictionary['sma']
    angle_mat = dictionary['angle']
    
    R = [Rbegin if Rbegin is not None else width/2]
    Rend = Rend if Rend is not None else max(img.shape)/np.sqrt(2)
    
    while R[-1] < Rend:
        R.append(R[-1] + (step if step is not None else width/2) )
    R = np.array(R)
    
    Meanflux = []
    Scatflux = []
    R_all = np.array([])
    azimuthal_light_all = np.array([])
    azimuthal_angle_all = np.array([])
    azimuthal_mask_all = np.array([])
    
    for i in range(len(R)):
        iall = (sma_mat > R[i]-width/2) & (sma_mat <= R[i]+width/2)
        if np.sum(iall) == 0:
            Meanflux.append(np.nan)
            continue
            
        azimuthal_light = img[iall]
        azimuthal_angle = angle_mat[iall]
        azimuthal_mask = mask[iall]
        
        iunmask = (azimuthal_mask == 0)
        if np.sum(iunmask) == 0:
            Meanflux.append(np.nan)
            Scatflux.append(np.nan)
            continue
        
        R_all = np.append(R_all, np.array([R[i] for j in range(len(azimuthal_light))])  )
        azimuthal_light_all = np.append(azimuthal_light_all, azimuthal_light)
        azimuthal_angle_all = np.append(azimuthal_angle_all, azimuthal_angle)
        azimuthal_mask_all = np.append(azimuthal_mask_all, azimuthal_mask)
        
        Meanflux.append(np.nanmean(azimuthal_light[iunmask]))
        Scatflux.append(np.nanstd(azimuthal_light[iunmask]))
        
    Meanflux, Scatflux = np.array(Meanflux), np.array(Scatflux) 
    choose = Meanflux > -10000
    R = R[choose]
    Scatflux = Scatflux[choose]
    Meanflux == Meanflux[choose]
####
    prof = {"R": "pixels", 
          'flux': "ADUs", 
           'err_flux': "ADUs"}
    prof["R"] = R
    prof["flux"] = Meanflux
    prof["err_flux"] = Scatflux
####
    azimuthal = {
        "R": "pixels",
        "light": "ADUs", 
        "theta": "angle",
        "mask": "mask"}    

    azimuthal["R"] = R_all
    azimuthal["light"] = azimuthal_light_all
    azimuthal["theta"] = azimuthal_angle_all
    azimuthal["mask"] = azimuthal_mask_all
    
    
    return {"prof": prof, "azimuthal": azimuthal}

def calculate_perimeter(a,b):
    return math.pi * ( 3*(a+b) - math.sqrt( (3*a + b) * (a + 3*b) ) )

def find_minR_resolution(eps, resol):
    for jj in range(2, 100):
        zhou = calculate_perimeter(jj, jj*(1-eps))
        if zhou > resol:
            return jj

def Extract_radial(img=None, mask=None, width=None, num=None, 
                   xcen=None, ycen=None, eps=None, PA=None):
    if img is None:
        print("Extract_radial(img=None, mask=None, width=None, num=None, xcen=None, ycen=None, eps=None, PA=None)")
        return

    if 'fit' in img:
        img = fits.open(img)[0].data
    if (mask is not None) and ('fit' in mask):
        mask = fits.open(mask)[0].data
    
    mask = np.zeros_like(img) if mask is None else mask
    mask_wedge = np.zeros_like(img)
    
    dictionary = dist_ellipse(shape=img.shape, xcen=xcen, ycen=ycen, eps=eps, PA=PA)
    sma_mat = dictionary['sma']
    angle_mat = dictionary['angle']
    
    minR_resolution = find_minR_resolution(eps, 360/width)
    theta_all = np.array([])
    radial_light_all = np.array([])
    radial_r_all     = np.array([])
    radial_mask_all  = np.array([])
    
    theta = np.linspace(0, 360 * (1 - 1.0 / num), num)
    for i in range(0, len(theta)):
        theta_low = theta[i] - width/2
        theta_hig = theta[i] + width/2
        
        k_low = (np.logical_and(angle_mat>theta_low, angle_mat<=theta[i])) \
            if theta_low > 0 else ( np.logical_or(angle_mat<=theta[i], angle_mat>theta_low+360) )
        k_hig = (np.logical_and(angle_mat<=theta_hig, angle_mat>=theta[i])) \
            if theta_hig < 360 else ( np.logical_or(angle_mat>=theta[i], angle_mat<=theta_hig-360) )
        kwant_1 = np.logical_or(k_low, k_hig)
        kwant = np.logical_and(kwant_1, sma_mat>minR_resolution)
        
        mask_wedge[kwant] = 1
        radial_light = img[kwant]
        radial_r     = sma_mat[kwant]
        radial_mask  = mask[kwant]
        
        theta_all = np.append(theta_all, np.array([theta[i] for j in range(len(radial_light))])  )
        radial_light_all = np.append(radial_light_all, radial_light)
        radial_r_all = np.append(radial_r_all, radial_r)
        radial_mask_all = np.append(radial_mask_all, radial_mask)
        
    radial = {
        "theta": "degree",
        "light": "ADUs", 
        "R": "pixels",
        "mask": "mask"}    

    radial["theta"] = theta_all
    radial["light"] = radial_light_all
    radial["R"] = radial_r_all
    radial["mask"] = radial_mask_all
    
    return {'mask_wedge': mask_wedge, 'radial':radial, 'angle':angle_mat, 'sma':sma_mat}







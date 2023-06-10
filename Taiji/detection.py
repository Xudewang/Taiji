import matplotlib.pyplot as plt
import numpy as np
import scarlet
import sep
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.table import Column, Table
from astropy.units import Quantity
from kuaizi.mock import Data
from scarlet.wavelet import Starlet

from .imtools import (_image_gaia_stars_tigress, extract_obj, image_gaia_stars,
                      increase_mask_regions, seg_remove_cen_obj,
                      seg_remove_obj)

# Some basic functions is from Kuaizi written by Jiaxuan Li.


def interpolate(data_lr, data_hr):
    ''' Interpolate low resolution data to high resolution

    Parameters
    ----------
    data_lr: Data
        low resolution Data
    data_hr: Data
        high resolution Data

    Result
    ------
    interp: numpy array
        the images in data_lr interpolated to the grid of data_hr
    '''
    frame_lr = scarlet.Frame(data_lr.images.shape,
                             wcs=data_lr.wcs,
                             channels=data_lr.channels)
    frame_hr = scarlet.Frame(data_hr.images.shape,
                             wcs=data_hr.wcs,
                             channels=data_hr.channels)

    coord_lr0 = (np.arange(data_lr.images.shape[1]),
                 np.arange(data_lr.images.shape[1]))
    coord_hr = (np.arange(data_hr.images.shape[1]),
                np.arange(data_hr.images.shape[1]))
    coord_lr = scarlet.resampling.convert_coordinates(coord_lr0, frame_lr,
                                                      frame_hr)

    interp = []
    for image in data_lr.images:
        interp.append(
            scarlet.interpolation.sinc_interp(image[None, :, :],
                                              coord_hr,
                                              coord_lr,
                                              angle=None)[0].T)
    return np.array(interp)


# Vanilla detection: SEP


def vanilla_detection(detect_image,
                      wcs_img=None,
                      mask=None,
                      sigma=3,
                      b=64,
                      f=3,
                      minarea=5,
                      convolve=False,
                      conv_radius=None,
                      deblend_nthresh=30,
                      deblend_cont=0.001,
                      sky_subtract=True,
                      show_fig=True,
                      **kwargs):
    '''
    Source detection using Source Extractor (actually SEP).

    Parameters
    ----------
    detect_image: 2-D numpy array
        image
    mask: numpy 2-D array
        image mask
    sigma: float
        detection threshold
    b: float
        box size
    f: float
        kernel size
    minarea: float
        minimum area for a source
    sky_subtract: bool
        whether subtract the estimated sky from the input image, then detect sources
    show_fig: bool
        whether plot a figure showing objects and segmentation map
    **kwargs: see `utils.extract_obj`.

    Result
    ------
    obj_cat: `astropy.table.Table` object
        catalog of detected sources
    segmap: numpy array
        segmentation map
    fig: `matplotlib.pyplot.figure` object
    '''
    result = extract_obj(detect_image,
                         mask=mask,
                         b=b,
                         f=f,
                         sigma=sigma,
                         minarea=minarea,
                         deblend_nthresh=deblend_nthresh,
                         deblend_cont=deblend_cont,
                         sky_subtract=sky_subtract,
                         convolve=convolve,
                         conv_radius=conv_radius,
                         show_fig=show_fig,
                         **kwargs)

    obj_cat = result[0]
    ra, dec = wcs_img.wcs_pix2world(list(zip(x, y)), 1).T
    arg_ind = obj_cat.argsort('flux', reverse=True)
    obj_cat.sort('flux', reverse=True)
    obj_cat['index'] = np.arange(len(obj_cat))
    segmap = result[1]
    segmap = np.append(-1, np.argsort(arg_ind))[segmap] + 1

    if show_fig is True:
        fig = result[2]
        return obj_cat, segmap, fig
    else:
        return obj_cat, segmap


def wavelet_detection(detect_image,
                      mask=None,
                      wavelet_lvl=4,
                      low_freq_lvl=0,
                      high_freq_lvl=1,
                      sigma=3,
                      b=64,
                      f=3,
                      minarea=5,
                      convolve=False,
                      conv_radius=None,
                      deblend_nthresh=30,
                      deblend_cont=0.001,
                      sky_subtract=True,
                      show_fig=True,
                      **kwargs):
    '''
    Perform wavelet transform before detecting sources. This enable us to emphasize features with high frequency or low frequency.

    Parameters
    ----------
    detect_image: 2-D numpy array
        image
    mask: numpy 2-D array
        image mask
    wavelet_lvl: int
        the number of wavelet decompositions
    high_freq_lvl: int
        this parameter controls how much low-frequency features are wiped away. It should be smaller than `wavelet_lvl - 1`.
        `high_freq_lvl=0` means no low-freq features are wiped (equivalent to vanilla), higher number yields a image with less low-freq features.
    sigma: float
        detection threshold
    b: float
        box size
    f: float
        kernel size
    minarea: float
        minimum area for a source
    sky_subtract: bool
        whether subtract the estimated sky from the input image, then detect sources
    show_fig: bool
        whether plot a figure showing objects and segmentation map
    **kwargs: see `utils.extract_obj`.

    Result
    ------
    obj_cat: `astropy.table.Table` object
        catalog of detected sources
    segmap: numpy array
        segmentation map
    fig: `matplotlib.pyplot.figure` object

    '''
    Sw = Starlet.from_image(detect_image)  # wavelet decomposition
    # Now the number of levels are calculated automatically
    # Can be accessed as lvl = Sw.scales
    w = Sw.coefficients
    iw = Sw.image

    if high_freq_lvl != 0:
        w[(high_freq_lvl):, :, :] = 0  # remove low frequency features
        # w: from high to low

    if low_freq_lvl != 0:
        w[:(low_freq_lvl), :, :] = 0  # remove high frequency features

    # image with high-frequency features highlighted
    high_freq_image = Starlet.from_coefficients(w).image

    result = vanilla_detection(high_freq_image,
                               mask=mask,
                               sigma=sigma,
                               b=b,
                               f=f,
                               minarea=minarea,
                               deblend_nthresh=deblend_nthresh,
                               deblend_cont=deblend_cont,
                               sky_subtract=sky_subtract,
                               convolve=convolve,
                               conv_radius=conv_radius,
                               show_fig=show_fig,
                               **kwargs)

    if show_fig is True:
        obj_cat, segmap, fig = result
        return obj_cat, segmap, fig
    else:
        obj_cat, segmap = result
        return obj_cat, segmap


def makeCatalog(datas,
                mask=None,
                lvl=3,
                method='wavelet',
                convolve=False,
                conv_radius=5,
                match_gaia=True,
                show_fig=True,
                visual_gaia=True,
                tigress=False,
                layer_ind=None,
                **kwargs):
    ''' Creates a detection catalog by combining low and high resolution data.

    This function is used for detection before running scarlet.
    It is particularly useful for stellar crowded fields and for detecting high frequency features.

    Parameters
    ----------
    datas: array
        array of Data objects
    mask: numpy 2-D array
        image mask
    lvl: int
        detection lvl, i.e., sigma in SEP
    method: str
        Options:
            "wavelet" uses wavelet decomposition of images before combination, emphasizes high-frequency features
            "vanilla" directly detect objects using SEP
    match_gaia: bool
        whether matching the detection catalog with Gaia dataset
    show_fig: bool
        whether show the detection catalog as a figure
    visual_gaia: bool
        whether mark Gaia stars in the figure
    kwargs:
        See the arguments of 'utils.extract_obj'.

    Returns
    -------
    obj_cat: `astropy.table.Table` object
        catalog of detected sources
    segmap: numpy array
        segmentation map
    bg_rms: array
        background level for each dataset
    '''
    if 'logger' in kwargs:
        logger = kwargs['logger']
    else:
        logger = None

    if len(datas) == 1:
        hr_images = datas[0].images / \
            np.abs(np.sum(datas[0].images, axis=(1, 2)))[:, None, None]
        # Detection image as the sum over all images
        detect_image = np.sum(hr_images, axis=0)
        # _weights = datas[0].weights.sum(axis=(1, 2)) / datas[0].weights.sum()
        # detect_image = (_weights[:, None, None] * datas[0].images).sum(axis=0)
    else:
        data_lr, data_hr = datas
        # Create observations for each image
        # Interpolate low resolution to high resolution
        interp = interpolate(data_lr, data_hr)
        # Normalisation of the interpolate low res images
        interp = interp / np.sum(interp, axis=(1, 2))[:, None, None]
        # Normalisation of the high res data
        hr_images = data_hr.images / \
            np.sum(data_hr.images, axis=(1, 2))[:, None, None]
        # Detection image as the sum over all images
        detect_image = np.sum(interp, axis=0) + np.sum(hr_images, axis=0)
        detect_image *= np.sum(data_hr.images)

    if np.size(detect_image.shape) == 3:
        detect = detect_image.mean(axis=0)
    else:
        detect = detect_image

    if layer_ind is not None:
        detect = datas[0].images[layer_ind]

    # we better subtract background first, before convolve
    result = vanilla_detection(detect,
                               mask=mask,
                               sigma=lvl,
                               show_fig=show_fig,
                               convolve=convolve,
                               conv_radius=conv_radius,
                               **kwargs)

    obj_cat = result[0]
    segmap = result[1]

    # RA and Dec
    if len(datas) == 1:
        ra, dec = datas[0].wcs.wcs_pix2world(obj_cat['x'], obj_cat['y'], 0)
        obj_cat.add_columns(
            [Column(data=ra, name='ra'),
             Column(data=dec, name='dec')])
    else:
        ra_lr, dec_lr = data_lr.wcs.wcs_pix2world(obj_cat['x'], obj_cat['y'],
                                                  0)
        ra_hr, dec_hr = data_hr.wcs.wcs_pix2world(obj_cat['x'], obj_cat['y'],
                                                  0)
        obj_cat.add_columns([
            Column(data=ra_lr, name='ra_lr'),
            Column(data=dec_lr, name='dec_lr')
        ])
        obj_cat.add_columns([
            Column(data=ra_hr, name='ra_hr'),
            Column(data=dec_hr, name='dec_hr')
        ])

    # Reorder columns
    colnames = obj_cat.colnames
    for item in ['dec', 'ra', 'y', 'x', 'index']:
        if item in colnames:
            colnames.remove(item)
            colnames.insert(0, item)
    obj_cat = obj_cat[colnames]
    obj_cat.add_column(Column(data=[None] * len(obj_cat), name='obj_type'),
                       index=0)

    # if len(datas) == 1:
    #     bg_rms = mad_wavelet(detect)
    # else:
    #     bg_rms = []
    #     for data in datas:
    #         bg_rms.append(mad_wavelet(detect))

    if match_gaia:
        obj_cat.add_column(
            Column(data=[None] * len(obj_cat), name='gaia_coord'))
        if len(datas) == 1:
            w = datas[0].wcs
            pixel_scale = w.to_header()['PC2_2'] * 3600
        else:
            w = data_hr.wcs
            pixel_scale = w.to_header()['PC2_2'] * 3600

        # Retrieve GAIA catalog
        if tigress:
            gaia_stars = _image_gaia_stars_tigress(detect,
                                                   w,
                                                   pixel_scale=pixel_scale,
                                                   verbose=True,
                                                   visual=visual_gaia,
                                                   logger=logger)
        else:
            gaia_stars = image_gaia_stars(detect,
                                          w,
                                          pixel_scale=pixel_scale,
                                          verbose=True,
                                          visual=visual_gaia)

        # Cross-match with SExtractor catalog
        from astropy.coordinates import SkyCoord, match_coordinates_sky
        temp, dist, _ = match_coordinates_sky(SkyCoord(ra=gaia_stars['ra'],
                                                       dec=gaia_stars['dec'],
                                                       unit='deg'),
                                              SkyCoord(ra=obj_cat['ra'],
                                                       dec=obj_cat['dec'],
                                                       unit='deg'),
                                              nthneighbor=1)
        flag = dist < 5 * u.arcsec
        star_mag = gaia_stars['phot_g_mean_mag'].data
        psf_ind = temp[flag]
        star_mag = star_mag[flag]
        bright_star_flag = star_mag < 19.0
        obj_cat['obj_type'][
            psf_ind[bright_star_flag]] = scarlet.source.ExtendedSource
        obj_cat['obj_type'][
            psf_ind[~bright_star_flag]] = scarlet.source.PointSource
        # we also use the coordinates from Gaia for bright stars
        obj_cat['gaia_coord'][psf_ind] = np.array(gaia_stars[['ra',
                                                              'dec']])[flag]

        # Cross-match for a second time: to deal with splitted bright stars
        temp_cat = obj_cat.copy(copy_data=True)
        temp_cat.remove_rows(psf_ind)
        temp2, dist2, _ = match_coordinates_sky(SkyCoord(ra=gaia_stars['ra'],
                                                         dec=gaia_stars['dec'],
                                                         unit='deg'),
                                                SkyCoord(ra=temp_cat['ra'],
                                                         dec=temp_cat['dec'],
                                                         unit='deg'),
                                                nthneighbor=1)
        flag2 = dist2 < 1 * u.arcsec
        psf_ind2 = temp_cat[temp2[flag2]]['index'].data
        # we also use the coordinates from Gaia for bright stars
        obj_cat.remove_rows(psf_ind2)
        # obj_cat['gaia_coord'][psf_ind2] = np.array(gaia_stars[['ra', 'dec']])[flag2]
        # obj_cat['obj_type'][psf_ind2] = scarlet.source.PointSource
        if logger:
            logger.info(f'    Matched {len(psf_ind)} stars from GAIA')
        print(f'    Matched {len(psf_ind)} stars from GAIA')

    obj_cat['index'] = np.arange(len(obj_cat))

    # Visualize the results
    if show_fig and match_gaia:
        from matplotlib.patches import Ellipse as mpl_ellip

        from .display import GRN, ORG

        fig = result[2]
        ax1 = fig.get_axes()[0]
        xlim = ax1.get_xlim()
        ylim = ax1.get_ylim()
        # Plot an ellipse for each object
        for star in gaia_stars[flag]:
            smask = mpl_ellip(xy=(star['x_pix'], star['y_pix']),
                              width=(2.0 * star['rmask_arcsec'] / pixel_scale),
                              height=(2.0 * star['rmask_arcsec'] /
                                      pixel_scale),
                              angle=0.0)
            smask.set_facecolor(ORG(0.2))
            smask.set_edgecolor(ORG(1.0))
            smask.set_alpha(0.3)
            ax1.add_artist(smask)

        # Show stars
        ax1.scatter(gaia_stars['x_pix'],
                    gaia_stars['y_pix'],
                    color=GRN(1.0),
                    s=100,
                    alpha=0.9,
                    marker='+')
        ax1.set_xlim(xlim)
        ax1.set_ylim(ylim)

    return obj_cat, segmap, 0  # bg_rms


def extract_centermain_object(obj_cat,
                              segmap,
                              image_data,
                              wcs_img,
                              size_around=30,
                              pixel_scale=0.168):
    """Remove the center object from the segmap and extract the enter information. The basic strategy is to find the brightest object around the center region (within size_around, default is 30 pixels).

    Args:
        obj_cat (_type_): _description_
        segmap (_type_): _description_
        image_data (_type_): _description_
        wcs_img (_type_): _description_
        size_around (int, optional): _description_. Defaults to 30.
        pixel_scale (float, optional): _description_. Defaults to 0.168.

    Returns:
        _type_: _description_
    """
    import copy

    # Generate manual measurement table
    obj_table = obj_cat
    a_arcsec, b_arcsec = (pixel_scale * obj_table['a']), (
        pixel_scale * obj_table['b'])  # arcsec
    x = obj_table['x']
    y = obj_table['y']
    ra, dec = wcs_img.wcs_pix2world(list(zip(x, y)), 1).T
    x2 = obj_table['x2']
    y2 = obj_table['y2']
    xy = obj_table['xy']
    a = obj_table['a']  # pixel
    b = obj_table['b']  # pixel
    theta = obj_table['theta']
    flux = obj_table['flux']
    R50_pixel = obj_table['R50']
    kron_rad_pixel = obj_table['kron_rad']
    index = obj_table['index']
    fwhm = obj_table['fwhm_custom']
    point_source = [((b_arcsec[i] / a_arcsec[i] > .9) and (a_arcsec[i] < .35))
                    for i in range(len(obj_table))]
    detection_cat = Table(
        [
            index, ra, dec, x, y, x2, y2, xy, a, b, a_arcsec, b_arcsec, theta,
            flux, R50_pixel, kron_rad_pixel, fwhm, point_source
        ],
        names=('index', 'ra', 'dec', 'x', 'y', 'x2', 'y2', 'xy', 'a', 'b',
               'a_arcsec', 'b_arcsec', 'theta', 'flux', 'R50_pixel',
               'kron_rad_pixel', 'fwhm_custom', 'point_source'),
        meta={'name': 'object table'})

    center_galaxy_labels = np.where(
        np.logical_and(
            np.logical_and(
                detection_cat['x'] > image_data.shape[0] / 2 - size_around,
                detection_cat['x'] < image_data.shape[0] / 2 + size_around),
            np.logical_and(
                detection_cat['y'] > image_data.shape[1] / 2 - size_around,
                detection_cat['y'] <
                image_data.shape[1] / 2 + size_around)))[0]

    galaxy_fluxes = np.array([
        detection_cat[np.where(detection_cat['index'] == label)]['flux'][0]
        for label in center_galaxy_labels
    ])
    main_galaxy_label = center_galaxy_labels[np.argmax(galaxy_fluxes)]
    info_center = detection_cat[main_galaxy_label]

    # construct the segmap without center.
    seg_copy = copy.deepcopy(segmap)
    seg_copy[segmap == segmap[int(info_center['x']),
                              int(info_center['y'])]] = 0

    return seg_copy, info_center


def remove_overlap(obj_cat,
                   segmap,
                   obj_cat_ori,
                   segmap_ori,
                   dist_minimum=1,
                   pixel_scale=0.168):
    """Remove overlapping objects combined with some other small criteria.

    Args:
        obj_cat (_type_): _description_
        segmap (_type_): _description_
        obj_cat_ori (_type_): _description_
        segmap_ori (_type_): _description_
        dist_minimum (int, optional): _description_. Defaults to 1.
        pixel_scale (float, optional): _description_. Defaults to 0.168.

    Returns:
        _type_: _description_
    """
    # Don't mask out objects that fall in the segmap of the central object
    import copy
    segmap = copy.deepcopy(segmap)
    segmap_ori = copy.deepcopy(segmap_ori)

    obj_cat = copy.deepcopy(obj_cat)
    obj_cat_ori = copy.deepcopy(obj_cat_ori)

    # overlap_flag is for objects which fall in the footprint
    # of central galaxy in the fist SEP detection
    # calculate the cen_indx_ori
    dist_ori = np.sqrt((obj_cat_ori['x'] - segmap_ori.shape[0] // 2)**2 +
                       (obj_cat_ori['y'] - segmap_ori.shape[1] // 2)**2)
    cen_indx_ori = np.argmin(dist_ori)

    dist = np.sqrt((obj_cat['x'] - segmap.shape[0] // 2)**2 +
                   (obj_cat['y'] - segmap.shape[1] // 2)**2)
    cen_indx = np.argmin(dist)

    # first we should remove the center object from the latter hot segmap.

    overlap_flag = [(segmap_ori == (cen_indx_ori + 1))[item] for item in list(
        zip(obj_cat['y'].astype(int), obj_cat['x'].astype(int)))]
    overlap_flag = np.array(overlap_flag)

    # overlap_flag = np.logical_and(overlap_flag,
    #                               dist <= 3 * obj_cat_ori[cen_indx_ori]['a'])
    overlap_flag = np.logical_and(
        overlap_flag,
        obj_cat['flux'] < 0.8 * obj_cat_ori[cen_indx_ori]['flux'])
    # overlap_flag = np.logical_and(overlap_flag,
    #                               obj_cat['b'] / obj_cat['a'] <= 0.8)
    overlap_flag |= (dist < dist_minimum / pixel_scale)

    for ind in np.where(overlap_flag)[0]:
        segmap[segmap == ind + 1] = 0

    return segmap


def divide_dilate_segmap(segmap_ori,
                         obj_cat,
                         divide_radius=50,
                         dilation_inner=1.5,
                         dilation_outer=3,
                         inner_ellipse_Na=3,
                         seeing_fwhm=0.65,
                         pixel_scale=0.168):
    """We should divide the segmap into two parts, one is for the central object, the other is for the rest objects. For different parts we use different dilation radius.

    Args:
        segmap_ori (_type_): _description_
        obj_cat (_type_): _description_
        divide_radius (int, optional): _description_. Defaults to 50.
        dilation_inner (float, optional): _description_. Defaults to 1.5.
        dilation_outer (int, optional): _description_. Defaults to 3.
        seeing_fwhm (float, optional): _description_. Defaults to 0.65.
    """
    import copy

    import sep

    segmap_inner = copy.deepcopy(segmap_ori)
    segmap_outer = copy.deepcopy(segmap_ori)
    obj_cagt = copy.deepcopy(obj_cat)
    dist = np.sqrt((obj_cat['x'] - segmap_ori.shape[0] // 2)**2 +
                   (obj_cat['y'] - segmap_ori.shape[1] // 2)**2)

    print('divide radius: ', divide_radius)

    for idx, obj in enumerate(obj_cat):
        if dist[idx] <= divide_radius:
            segmap_outer[segmap_outer == (obj['index'] + 1)] = 0
        elif dist[idx] > divide_radius:
            segmap_inner[segmap_inner == (obj['index'] + 1)] = 0

    # dilate the segmap_inner and segmap_outer
    segmap_inner_dilation = increase_mask_regions(
        segmap_inner,
        method='gaussian',
        size=(dilation_inner * seeing_fwhm / pixel_scale))
    segmap_outer_dilation = increase_mask_regions(
        segmap_outer,
        method='gaussian',
        size=(dilation_outer * seeing_fwhm / pixel_scale))

    # we can construct the the inner segmap with 3 times of the a and b of the central object using sep make_ellipse function.
    maskEllipse_x_arr = []
    maskEllipse_y_arr = []
    maskEllipse_a_arr = []
    maskEllipse_b_arr = []
    maskEllipse_theta_arr = []
    for idx, obj in enumerate(obj_cat):
        if dist[idx] <= divide_radius:
            maskEllipse_x_arr.append(obj['x'])
            maskEllipse_y_arr.append(obj['y'])
            maskEllipse_a_arr.append(obj['a'])
            maskEllipse_b_arr.append(obj['b'])
            maskEllipse_theta_arr.append(obj['theta'])

    mask_ellipse_inner = np.zeros(segmap_ori.shape, dtype=np.bool)
    sep.mask_ellipse(mask_ellipse_inner,
                     maskEllipse_x_arr,
                     maskEllipse_y_arr,
                     maskEllipse_a_arr,
                     maskEllipse_b_arr,
                     maskEllipse_theta_arr,
                     r=inner_ellipse_Na)

    return segmap_inner_dilation, segmap_outer_dilation, mask_ellipse_inner


def segmap_coldhot_removeinnermost(obj_cat_cold,
                                   seg_cold,
                                   obj_cat_hot,
                                   seg_hot,
                                   dist_inner_criteria=3,
                                   q_criteria=0.8,
                                   dilate_radius_criteria=6,
                                   dist_unit_flag='r50',
                                   dilation_inner=1.5,
                                   dilation_outer=3,
                                   seeing_fwhm=0.65,
                                   image_data=None,
                                   inner_mask='segmap',
                                   inner_ellipse_Na=3,
                                   show_removeinnermost_img_direct=False,
                                   show_removeinnermost_img_dilated=False,
                                   show_removeinnermost_img_parts=False):
    # combine the segmap of cold and hot mode. The basic spirit is to remove the central object from the hot mode segmap, and then add the cold mode segmap to the hot mode segmap. But we need to make sure that the segmap of hot mode does not divide the main object in the center several pieces in the cold mode. If it does, we need to remove the small pieces in the hot mode segmap. The selection criteria is the axis ratio of the small pieces should be larger than for example 0.25.

    # remove the central object from the hot and cold mode segmap
    import copy

    from astropy.visualization import simple_norm

    seg_hot = copy.deepcopy(seg_hot)
    seg_cold = copy.deepcopy(seg_cold)

    obj_cat_cold = Table(obj_cat_cold)
    obj_cat_hot = Table(obj_cat_hot)

    cen_obj_idx_cold = np.argmin(
        (obj_cat_cold['x'] - seg_cold.shape[1] // 2)**2 +
        (obj_cat_cold['y'] - seg_cold.shape[0] // 2)**2)

    cen_obj_idx_hot = np.argmin((obj_cat_hot['x'] -
                                 seg_cold.shape[1] // 2)**2 +
                                (obj_cat_hot['y'] - seg_cold.shape[0] // 2)**2)

    info_cen_obj_cold = obj_cat_cold[cen_obj_idx_cold]
    a_cen_cold = info_cen_obj_cold['a']
    #print('a_cen_cold', a_cen_cold)
    b_cen_cold = info_cen_obj_cold['b']
    #print('b_cen_cold', b_cen_cold)
    kronrad_cen_cold = info_cen_obj_cold['kron_rad']
    #print('kron rad cold: ', kronrad_cen_cold)
    r50_cen_cold = info_cen_obj_cold['R50']
    r90_cen_cold = info_cen_obj_cold['R90']
    flux_cen_cold = info_cen_obj_cold['flux']

    info_cen_obj_hot = obj_cat_hot[cen_obj_idx_hot]
    x_cen_obj_hot = info_cen_obj_hot['x']
    y_cen_obj_hot = info_cen_obj_hot['y']

    if dist_unit_flag == 'a':
        dist_unit = a_cen_cold
    elif dist_unit_flag == 'r50':
        dist_unit = r50_cen_cold
    elif dist_unit_flag == 'r90':
        dist_unit = r90_cen_cold
    elif dist_unit_flag == 'kronrad':
        dist_unit = kronrad_cen_cold
    print('dist unit: ', dist_unit)

    boundary_innermost_criteria = dist_inner_criteria * dist_unit

    idx_remove_arr = []
    seg_hot_removecenter = seg_remove_cen_obj(seg_hot)
    obj_cat_hot_remove = copy.deepcopy(obj_cat_hot)
    obj_cat_hot_remove.remove_row(cen_obj_idx_hot)

    for i, obj in enumerate(obj_cat_hot_remove):

        dist_cen_hot = np.sqrt((obj_cat_hot_remove[i]['x'] -
                                seg_hot_removecenter.shape[0] // 2)**2 +
                               (obj_cat_hot_remove[i]['y'] -
                                seg_hot_removecenter.shape[1] // 2)**2)

        if np.logical_and(np.logical_and(dist_cen_hot < boundary_innermost_criteria,
                          (obj['b'] / obj['a'] <= q_criteria)), obj['flux'] < 0.5 * flux_cen_cold):
            seg_hot_removecenter[seg_hot_removecenter == (
                obj_cat_hot_remove[i]['index'] + 1)] = 0
            idx_remove_arr.append(i)

    obj_cat_hot_remove.remove_rows(idx_remove_arr)
    #print('obj_cat_hot_remove: ', obj_cat_hot_remove)
    fig, ax = plt.subplots(figsize=(6, 6))
    plt.imshow(seg_hot_removecenter, origin='lower')
    plt.title('seg_hot_removecenter')

    # after removing the center object of the segmap_cold, we should get the segmap_cold_removecenter and obj_cat_cold_removecenter
    seg_cold_removecenter = seg_remove_cen_obj(seg_cold)

    obj_cat_cold_removecenter = copy.deepcopy(obj_cat_cold)
    obj_cat_cold_removecenter.remove_row(cen_obj_idx_cold)
    #print('obj_cat_cold_removecenter: ', obj_cat_cold_removecenter)
    fig, ax = plt.subplots(figsize=(6, 6))
    plt.imshow(seg_cold_removecenter,
               origin='lower',
               label='segmap_cold_removecenter')
    plt.title('seg_cold_removecenter')

    #add the cold mode segmap to the hot mode segmap
    seg_combine_direct = np.logical_or(seg_hot_removecenter,
                                       seg_remove_cen_obj(seg_cold))

    # dilate the segmap_cold_removecenter and segmap_hot_remove with different dilation parameters and then add them together
    seg_cold_removecenter_inner_dilation, seg_cold_removecenter_outer_dilation, maskEllipse_cold_inner = divide_dilate_segmap(
        seg_cold_removecenter,
        obj_cat_cold_removecenter,
        divide_radius=dilate_radius_criteria * r50_cen_cold,
        dilation_inner=dilation_inner,
        dilation_outer=dilation_outer,
        seeing_fwhm=seeing_fwhm,
        inner_ellipse_Na=inner_ellipse_Na)

    seg_hot_remove_inner_dilation, seg_hot_remove_outer_dilation, maskEllipse_hot_inner = divide_dilate_segmap(
        seg_hot_removecenter,
        obj_cat_hot_remove,
        divide_radius=dilate_radius_criteria * r50_cen_cold,
        dilation_inner=dilation_inner,
        dilation_outer=dilation_outer,
        seeing_fwhm=seeing_fwhm,
        inner_ellipse_Na=inner_ellipse_Na)

    # then we should combine the cold+hot dilation masks.
    seg_combine_inner_dilation = np.logical_or(
        seg_cold_removecenter_inner_dilation, seg_hot_remove_inner_dilation)
    seg_combine_outer_dilation = np.logical_or(
        seg_cold_removecenter_outer_dilation, seg_hot_remove_outer_dilation)
    maskEllipse_combine_inner = np.logical_or(maskEllipse_cold_inner,
                                              maskEllipse_hot_inner)

    if inner_mask == 'segmap':
        seg_combine_dilation = np.logical_or(seg_combine_inner_dilation,
                                             seg_combine_outer_dilation)
    elif inner_mask == 'ellipse':
        seg_combine_dilation = np.logical_or(seg_combine_outer_dilation,
                                             maskEllipse_combine_inner)

    # show the image data with segmap. And add the circular mask for the central object.
    if show_removeinnermost_img_direct:
        fig, ax = plt.subplots(figsize=(6, 6))

        norm = simple_norm(image_data, 'sqrt', percent=99.9)
        ax.imshow(image_data, norm=norm, origin='lower', cmap='Greys')
        ax.imshow(seg_combine_direct, origin='lower', alpha=0.5, cmap='Blues')
        ax.set_title('Direct removeinnermost combined Segmap')

        # add the cicurlar patch for the central object
        circle = plt.Circle((seg_hot.shape[0] // 2, seg_hot.shape[1] // 2),
                            boundary_innermost_criteria,
                            color='r',
                            fill=False)
        ax.add_artist(circle)

    if show_removeinnermost_img_dilated:

        fig, ax = plt.subplots(figsize=(6, 6))

        norm = simple_norm(image_data, 'sqrt', percent=99.9)
        ax.imshow(image_data, norm=norm, origin='lower', cmap='Greys')
        ax.imshow(seg_combine_dilation,
                  origin='lower',
                  alpha=0.5,
                  cmap='Blues')
        ax.set_title('Removeinnermost combined Segmap')

        # add the cicurlar patch for the central object
        circle = plt.Circle(
            (seg_hot.shape[0] // 2, seg_hot.shape[1] // 2),
            dilate_radius_criteria * r50_cen_cold,
            color='green',
            fill=False,
            label=f'dilation radius: {dilate_radius_criteria} $Re$',
            lw=2)
        ax.add_artist(circle)

        plt.legend()

    if show_removeinnermost_img_parts:
        fig, ax = plt.subplots(figsize=(6, 6))

        norm = simple_norm(image_data, 'sqrt', percent=99.9)
        ax.imshow(image_data, norm=norm, origin='lower', cmap='Greys')
        ax.imshow(seg_combine_inner_dilation,
                  origin='lower',
                  alpha=0.5,
                  cmap='Blues')
        ax.set_title('Inner dilation Segmap')

        # add the cicurlar patch for the central object
        circle = plt.Circle((seg_hot.shape[0] // 2, seg_hot.shape[1] // 2),
                            dilate_radius_criteria * r50_cen_cold,
                            color='green',
                            fill=False)
        ax.add_artist(circle)

        fig, ax = plt.subplots(figsize=(6, 6))

        norm = simple_norm(image_data, 'sqrt', percent=99.9)
        ax.imshow(image_data, norm=norm, origin='lower', cmap='Greys')
        ax.imshow(seg_combine_outer_dilation,
                  origin='lower',
                  alpha=0.5,
                  cmap='Blues')
        ax.set_title('Outer dilation Segmap')

        # add the cicurlar patch for the central object
        circle = plt.Circle((seg_hot.shape[0] // 2, seg_hot.shape[1] // 2),
                            dilate_radius_criteria * r50_cen_cold,
                            color='green',
                            fill=False)
        ax.add_artist(circle)

    return seg_combine_direct, seg_combine_dilation


def make_simple_coldhot_mask(image_data,
                               wcs_img = None,
                               b1=256,
                               f1=3,
                               sigma1=1.5,
                               b2=256,
                               f2=3,
                               sigma2=1.,
                               pixel_scale=0.168,
                               minarea=5,
                               deblend_nthresh1=32,
                               deblend_nthresh2=32,
                               deblend_cont1=0.1,
                               deblend_cont2=0.005,
                               sky_subtract=True,
                               show_segmap=False,
                               mask=None,
                               convolve=False,
                               conv_radius=3,
                               dist_inner_criteria=3,
                               dilate_radius_criteria=6,
                               dist_unit_flag='r50',
                               dilation_inner=1.5,
                               dilation_outer=3,
                               seeing_fwhm=0.65,
                               inner_mask='segmap',
                               inner_ellipse_Na=3,
                               show_removeinnermost_segmap=False,
                               show_removeinnermost_img_direct=False,
                               show_removeinnermost_img_dilated=False,
                               show_removeinnermost_img_parts=False,
                               save_mask_dilation=None,
                               save_mask_ori=None):
    # combine the segmap of cold and hot mode. The basic spirit is to remove the central object from the hot mode segmap, and then add the cold mode segmap to the hot mode segmap. But we need to make sure that the segmap of hot mode does not divide the main object in the center several pieces in the cold mode. If it does, we need to remove the small pieces in the hot mode segmap. The selection criteria is the axis ratio of the small pieces should be larger than for example 0.25.

    # remove the central object from the hot and cold mode segmap
    import copy

    from astropy.visualization import simple_norm

    from Taiji.detection import divide_dilate_segmap
    from Taiji.imtools import easy_saveData_Tofits, extract_obj, save_to_fits

    result_cold = extract_obj(image_data,
                              b=b1,
                              f=f1,
                              sigma=sigma1,
                              pixel_scale=pixel_scale,
                              minarea=minarea,
                              deblend_nthresh=deblend_nthresh1,
                              deblend_cont=deblend_cont1,
                              sky_subtract=sky_subtract,
                              show_fig=show_segmap,
                              mask=mask,
                              convolve=convolve,
                              conv_radius=conv_radius)

    obj_cat_cold = result_cold[0]
    seg_cold = result_cold[1]

    result_hot = extract_obj(image_data,
                             b=b2,
                             f=f2,
                             sigma=sigma2,
                             pixel_scale=pixel_scale,
                             minarea=minarea,
                             deblend_nthresh=deblend_nthresh2,
                             deblend_cont=deblend_cont2,
                             sky_subtract=sky_subtract,
                             show_fig=show_segmap,
                             mask=mask,
                             convolve=convolve,
                             conv_radius=6)

    obj_cat_hot = result_hot[0]
    seg_hot = result_hot[1]

    seg_hot = copy.deepcopy(seg_hot)
    seg_cold = copy.deepcopy(seg_cold)

    obj_cat_cold = Table(obj_cat_cold)
    obj_cat_hot = Table(obj_cat_hot)

    cen_obj_idx_cold = np.argmin(
        (obj_cat_cold['x'] - seg_cold.shape[1] // 2)**2 +
        (obj_cat_cold['y'] - seg_cold.shape[0] // 2)**2)

    cen_obj_idx_hot = np.argmin((obj_cat_hot['x'] -
                                 seg_cold.shape[1] // 2)**2 +
                                (obj_cat_hot['y'] - seg_cold.shape[0] // 2)**2)

    info_cen_obj_cold = obj_cat_cold[cen_obj_idx_cold]
    a_cen_cold = info_cen_obj_cold['a']
    #print('a_cen_cold', a_cen_cold)
    b_cen_cold = info_cen_obj_cold['b']
    #print('b_cen_cold', b_cen_cold)
    kronrad_cen_cold = info_cen_obj_cold['kron_rad']
    #print('kron rad cold: ', kronrad_cen_cold)
    r50_cen_cold = info_cen_obj_cold['R50']
    r90_cen_cold = info_cen_obj_cold['R90']
    flux_cen_cold = info_cen_obj_cold['flux']

    info_cen_obj_hot = obj_cat_hot[cen_obj_idx_hot]
    x_cen_obj_hot = info_cen_obj_hot['x']
    y_cen_obj_hot = info_cen_obj_hot['y']

    if dist_unit_flag == 'a':
        dist_unit = a_cen_cold
    elif dist_unit_flag == 'r50':
        dist_unit = r50_cen_cold
    elif dist_unit_flag == 'r90':
        dist_unit = r90_cen_cold
    elif dist_unit_flag == 'kronrad':
        dist_unit = kronrad_cen_cold
    print('dist unit: ', dist_unit)

    boundary_innermost_criteria = dist_inner_criteria * dist_unit

    idx_remove_arr = []
    seg_hot_removecenter = seg_remove_cen_obj(seg_hot)
    obj_cat_hot_remove = copy.deepcopy(obj_cat_hot)
    obj_cat_hot_remove.remove_row(cen_obj_idx_hot)

    if show_removeinnermost_segmap == True:
        fig, ax = plt.subplots(figsize=(6, 6))
        plt.imshow(seg_hot_removecenter, origin='lower')
        plt.title('seg_hot_removecenter')

    # after removing the center object of the segmap_cold, we should get the segmap_cold_removecenter and obj_cat_cold_removecenter
    seg_cold_removecenter = seg_remove_cen_obj(seg_cold)

    obj_cat_cold_removecenter = copy.deepcopy(obj_cat_cold)
    obj_cat_cold_removecenter.remove_row(cen_obj_idx_cold)
    #print('obj_cat_cold_removecenter: ', obj_cat_cold_removecenter)
    if show_removeinnermost_segmap == True:
        fig, ax = plt.subplots(figsize=(6, 6))
        plt.imshow(seg_cold_removecenter,
                origin='lower',
                label='segmap_cold_removecenter')
        plt.title('seg_cold_removecenter')

    #add the cold mode segmap to the hot mode segmap
    seg_combine_direct = np.logical_or(seg_hot_removecenter,
                                       seg_remove_cen_obj(seg_cold))

    # dilate the segmap_cold_removecenter and segmap_hot_remove with different dilation parameters and then add them together
    seg_cold_removecenter_inner_dilation, seg_cold_removecenter_outer_dilation, maskEllipse_cold_inner = divide_dilate_segmap(
        seg_cold_removecenter,
        obj_cat_cold_removecenter,
        divide_radius=dilate_radius_criteria * r50_cen_cold,
        dilation_inner=dilation_inner,
        dilation_outer=dilation_outer,
        seeing_fwhm=seeing_fwhm,
        inner_ellipse_Na=inner_ellipse_Na)

    seg_hot_remove_inner_dilation, seg_hot_remove_outer_dilation, maskEllipse_hot_inner = divide_dilate_segmap(
        seg_hot_removecenter,
        obj_cat_hot_remove,
        divide_radius=dilate_radius_criteria * r50_cen_cold,
        dilation_inner=dilation_inner,
        dilation_outer=dilation_outer,
        seeing_fwhm=seeing_fwhm,
        inner_ellipse_Na=inner_ellipse_Na)

    # then we should combine the cold+hot dilation masks.
    seg_combine_inner_dilation = np.logical_or(
        seg_cold_removecenter_inner_dilation, seg_hot_remove_inner_dilation)
    seg_combine_outer_dilation = np.logical_or(
        seg_cold_removecenter_outer_dilation, seg_hot_remove_outer_dilation)
    maskEllipse_combine_inner = np.logical_or(maskEllipse_cold_inner,
                                              maskEllipse_hot_inner)

    if inner_mask == 'segmap':
        seg_combine_dilation = np.logical_or(seg_combine_inner_dilation,
                                             seg_combine_outer_dilation)
    elif inner_mask == 'ellipse':
        seg_combine_dilation = np.logical_or(seg_combine_outer_dilation,
                                             maskEllipse_combine_inner)
        
    # save the mask 
    if save_mask_ori is not None:
        save_to_fits(img=seg_combine_direct.astype('float32'),
                 fits_file=save_mask_ori,
                 wcs=wcs_img)
        
    if save_mask_dilation is not None:
        save_to_fits(img=seg_combine_dilation.astype('float32'),
                 fits_file=save_mask_dilation,
                 wcs=wcs_img)

    # show the image data with segmap. And add the circular mask for the central object.
    if show_removeinnermost_img_direct:
        fig, ax = plt.subplots(figsize=(6, 6))

        norm = simple_norm(image_data, 'sqrt', percent=99.9)
        ax.imshow(image_data, norm=norm, origin='lower', cmap='Greys')
        ax.imshow(seg_combine_direct, origin='lower', alpha=0.5, cmap='Blues')
        ax.set_title('Direct removeinnermost combined Segmap')

        # add the cicurlar patch for the central object
        circle = plt.Circle((seg_hot.shape[0] // 2, seg_hot.shape[1] // 2),
                            boundary_innermost_criteria,
                            color='r',
                            fill=False)
        ax.add_artist(circle)

    if show_removeinnermost_img_dilated:

        fig, ax = plt.subplots(figsize=(6, 6))

        norm = simple_norm(image_data, 'sqrt', percent=99.9)
        ax.imshow(image_data, norm=norm, origin='lower', cmap='Greys')
        ax.imshow(seg_combine_dilation,
                  origin='lower',
                  alpha=0.5,
                  cmap='Blues')
        ax.set_title('Removeinnermost combined Segmap')

        # add the cicurlar patch for the central object
        circle = plt.Circle(
            (seg_hot.shape[0] // 2, seg_hot.shape[1] // 2),
            dilate_radius_criteria * r50_cen_cold,
            color='green',
            fill=False,
            label=f'dilation radius: {dilate_radius_criteria} $Re$',
            lw=2)
        ax.add_artist(circle)

        plt.legend()

    if show_removeinnermost_img_parts:
        fig, ax = plt.subplots(figsize=(6, 6))

        norm = simple_norm(image_data, 'sqrt', percent=99.9)
        ax.imshow(image_data, norm=norm, origin='lower', cmap='Greys')
        ax.imshow(seg_combine_inner_dilation,
                  origin='lower',
                  alpha=0.5,
                  cmap='Blues')
        ax.set_title('Inner dilation Segmap')

        # add the cicurlar patch for the central object
        circle = plt.Circle((seg_hot.shape[0] // 2, seg_hot.shape[1] // 2),
                            dilate_radius_criteria * r50_cen_cold,
                            color='green',
                            fill=False)
        ax.add_artist(circle)

        fig, ax = plt.subplots(figsize=(6, 6))

        norm = simple_norm(image_data, 'sqrt', percent=99.9)
        ax.imshow(image_data, norm=norm, origin='lower', cmap='Greys')
        ax.imshow(seg_combine_outer_dilation,
                  origin='lower',
                  alpha=0.5,
                  cmap='Blues')
        ax.set_title('Outer dilation Segmap')

        # add the cicurlar patch for the central object
        circle = plt.Circle((seg_hot.shape[0] // 2, seg_hot.shape[1] // 2),
                            dilate_radius_criteria * r50_cen_cold,
                            color='green',
                            fill=False)
        ax.add_artist(circle)

    return seg_combine_direct, seg_combine_dilation

def coldhot_detection(image_data,
                      b1=256,
                      f1=3,
                      sigma1=3.,
                      b2=64,
                      f2=3,
                      sigma2=1.,
                      pixel_scale=0.168,
                      minarea=5,
                      deblend_nthresh1=32,
                      deblend_nthresh2=32,
                      deblend_cont1=0.05,
                      deblend_cont2=0.001,
                      sky_subtract=True,
                      show_segmap=False,
                      mask=None,
                      convolve=False,
                      conv_radius=3,
                      dist_inner_criteria=3,
                      q_criteria=0.8,
                      dilate_radius_criteria=5,
                      dist_unit_flag='a',
                      dilation_inner=1.,
                      dilation_outer=3,
                      seeing_fwhm=0.65,
                      inner_mask='segmap',
                      inner_ellipse_Na=5,
                      show_removeinnermost_img_direct=False,
                      show_removeinnermost_img_dilated=False,
                      show_removeinnermost_img_parts=False):

    result_cold = extract_obj(image_data,
                              b=b1,
                              f=f1,
                              sigma=sigma1,
                              pixel_scale=pixel_scale,
                              minarea=minarea,
                              deblend_nthresh=deblend_nthresh1,
                              deblend_cont=deblend_cont1,
                              sky_subtract=sky_subtract,
                              show_fig=show_segmap,
                              mask=mask,
                              convolve=convolve,
                              conv_radius=conv_radius)

    obj_cat_cold = result_cold[0]
    segmap_cold = result_cold[1]

    result_hot = extract_obj(image_data,
                                          b=b2,
                                          f=f2,
                                          sigma=sigma2,
                                          pixel_scale=pixel_scale,
                                          minarea=minarea,
                                          deblend_nthresh=deblend_nthresh2,
                                          deblend_cont=deblend_cont2,
                                          sky_subtract=sky_subtract,
                                          show_fig=show_segmap,
                                          mask=mask,
                                          convolve=convolve,
                                          conv_radius=6)
    
    obj_cat_hot = result_hot[0]
    segmap_hot = result_hot[1]

    seg_removecenter_hot = seg_remove_cen_obj(segmap_hot)
    seg_removecenter_cold = seg_remove_cen_obj(segmap_cold)
    segmap_combine_direct = np.logical_or(seg_removecenter_cold,
                                          seg_removecenter_hot)

    segmap_combine_removeoverlap = np.logical_or(
        seg_remove_cen_obj(
            remove_overlap(obj_cat_hot,
                           segmap_hot,
                           obj_cat_cold,
                           segmap_cold,
                           dist_minimum=1.5*seeing_fwhm,
                           pixel_scale=pixel_scale)),
        seg_remove_cen_obj(segmap_cold))

    seg_combine_removeinnermost, seg_combine_removeinnermost_dilation = segmap_coldhot_removeinnermost(
        obj_cat_cold,
        segmap_cold,
        obj_cat_hot,
        segmap_hot,
        dist_inner_criteria=dist_inner_criteria,
        q_criteria=q_criteria,
        dilate_radius_criteria=dilate_radius_criteria,
        dist_unit_flag=dist_unit_flag,
        dilation_inner=dilation_inner,
        dilation_outer=dilation_outer,
        seeing_fwhm=seeing_fwhm,
        image_data=image_data,
        inner_mask=inner_mask,
        inner_ellipse_Na=inner_ellipse_Na,
        show_removeinnermost_img_direct=show_removeinnermost_img_direct,
        show_removeinnermost_img_dilated=show_removeinnermost_img_dilated,
        show_removeinnermost_img_parts=show_removeinnermost_img_parts)

    return seg_combine_removeinnermost_dilation, segmap_combine_removeoverlap, segmap_combine_direct
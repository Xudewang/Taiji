from __future__ import division, print_function
from matplotlib.colorbar import Colorbar
from matplotlib import rcParams
import os
import sys
import sep
import copy
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse as mpl_ellip
from contextlib import contextmanager

from astropy.io import fits
from astropy import wcs
from astropy.table import Table, Column, vstack
from astropy import units as u
from astropy.units import Quantity
from astropy.coordinates import SkyCoord

from .imtools import display_single, SEG_CMAP, ORG


@contextmanager
def suppress_stdout():
    """Suppress the output.

    Based on: https://thesmithfam.org/blog/2012/10/25/temporarily-suppress-console-output-in-python/
    """
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout
    
def set_matplotlib(style='default', usetex=False, fontsize=13, figsize=(6, 5), dpi=100):
    '''
    Default matplotlib settings, borrowed from Song Huang. I really like his plotting style.

    Parameters:
        style (str): options are "JL", "SM" (supermongo-like).
    '''
    # Use JL as a template
    if style == 'default':
        plt.style.use(os.path.join('/home/dewang/Taiji/', 'mplstyle/default.mplstyle'))
    else:
        plt.style.use(os.path.join('/home/dewang/Taiji/', 'mplstyle/JL.mplstyle'))
    rcParams.update({'font.size': fontsize,
                     'figure.figsize': "{0}, {1}".format(figsize[0], figsize[1]),
                     'text.usetex': usetex,
                     'figure.dpi': dpi})

    if style == 'DW':
        plt.style.use(['science', 'seaborn-colorblind'])

        plt.rcParams['figure.figsize'] = (10,7)
        plt.rcParams['font.size'] = 25
        plt.rcParams['lines.linewidth'] = 3
        plt.rcParams['xtick.labelsize'] = 25
        plt.rcParams['ytick.labelsize'] = 25
        plt.rcParams['xtick.direction'] = 'in'
        plt.rcParams['ytick.direction'] = 'in'
        plt.rcParams['xtick.major.size'] = 8
        plt.rcParams['ytick.major.size'] = 8
        plt.rcParams['xtick.minor.size'] = 5
        plt.rcParams['ytick.minor.size'] = 5
        plt.rcParams['xtick.major.pad'] = 5
        plt.rcParams['xtick.minor.pad'] = 4.8
        plt.rcParams['ytick.major.pad'] = 5
        plt.rcParams['ytick.minor.pad'] = 4.8
        plt.rcParams['xtick.top'] = True
        plt.rcParams['ytick.right'] = True
        plt.rcParams['legend.frameon'] = True
        plt.rcParams['axes.labelpad'] = 8.0
        plt.rcParams['figure.constrained_layout.h_pad'] = 0
        plt.rcParams['text.usetex'] = True
        plt.rc('text', usetex=True)
        plt.rcParams['font.sans-serif'] = ['Times New Roman']
        plt.tick_params(axis='both', which='minor', labelsize=18)
        plt.rcParams['legend.edgecolor'] = 'black'
        # plt.rcParams['xtick.major.width'] = 3.8
        # plt.rcParams['xtick.minor.width'] = 3.2
        # plt.rcParams['ytick.major.width'] = 3.8 
        # plt.rcParams['ytick.minor.width'] = 3.2
        # plt.rcParams['axes.linewidth'] = 5
        import matplotlib.ticker
        from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                                       AutoMinorLocator)
        plt.close()

    if style == 'nature':
        rcParams.update({
            "font.family": "sans-serif",
            # The default edge colors for scatter plots.
            "scatter.edgecolors": "black",
            "mathtext.fontset": "stixsans"
        })
       
def extract_obj(img, mask=None, b=64, f=3, sigma=5, pixel_scale=0.168, minarea=5,
                convolve=False, conv_radius=None,
                deblend_nthresh=32, deblend_cont=0.005, clean_param=1.0,
                sky_subtract=False, flux_auto=True, flux_aper=None, show_fig=False,
                verbose=True, logger=None):
    '''
    Extract objects for a given image using ``sep`` (a Python-wrapped ``SExtractor``). 
    For more details, please check http://sep.readthedocs.io and documentation of SExtractor.

    Parameters:
        img (numpy 2-D array): input image
        mask (numpy 2-D array): image mask
        b (float): size of box
        f (float): size of convolving kernel
        sigma (float): detection threshold
        pixel_scale (float): default is 0.168 (HSC pixel size). This only affect the figure scale bar.
        minarea (float): minimum number of connected pixels
        deblend_nthresh (float): Number of thresholds used for object deblending
        deblend_cont (float): Minimum contrast ratio used for object deblending. Set to 1.0 to disable deblending. 
        clean_param (float): Cleaning parameter (see SExtractor manual)
        sky_subtract (bool): whether subtract sky before extract objects (this will affect the measured flux).
        flux_auto (bool): whether return AUTO photometry (see SExtractor manual)
        flux_aper (list): such as [3, 6], which gives flux within [3 pix, 6 pix] annulus.

    Returns:
        :
            objects: `astropy` Table, containing the positions,
                shapes and other properties of extracted objects.

            segmap: 2-D numpy array, segmentation map
    '''
    import sep
    # Subtract a mean sky value to achieve better object detection
    b = b  # Box size
    f = f  # Filter width

    # if convolve:
    #     from astropy.convolution import convolve, Gaussian2DKernel
    #     img = convolve(img.astype(float), Gaussian2DKernel(conv_radius))

    try:
        bkg = sep.Background(img, bw=b, bh=b, fw=f, fh=f)
    except ValueError as e:
        img = img.byteswap().newbyteorder()
        bkg = sep.Background(img, bw=b, bh=b, fw=f, fh=f)

    data_sub = img - bkg.back()

    sigma = sigma

    if sky_subtract:
        input_data = data_sub
    else:
        input_data = img

    if convolve:
        from astropy.convolution import convolve, Gaussian2DKernel
        input_data = convolve(input_data.astype(
            float), Gaussian2DKernel(conv_radius))
        bkg = sep.Background(input_data, bw=b, bh=b, fw=f, fh=f)
        input_data -= bkg.globalback

    objects, segmap = sep.extract(input_data,
                                  sigma,
                                  mask=mask,
                                  err=bkg.globalrms,
                                  segmentation_map=True,
                                  filter_type='matched',
                                  deblend_nthresh=deblend_nthresh,
                                  deblend_cont=deblend_cont,
                                  clean=True,
                                  clean_param=clean_param,
                                  minarea=minarea)

    if verbose:
        if logger is not None:
            logger.info("    Detected %d objects" % len(objects))
        print("    Detected %d objects" % len(objects))
    objects = Table(objects)
    objects.add_column(Column(data=np.arange(len(objects)), name='index'))
    # Maximum flux, defined as flux within 6 * `a` (semi-major axis) in radius.
    objects.add_column(Column(data=sep.sum_circle(input_data, objects['x'], objects['y'],
                                                  6. * objects['a'])[0], name='flux_max'))
    # Add FWHM estimated from 'a' and 'b'.
    # This is suggested here: https://github.com/kbarbary/sep/issues/34
    objects.add_column(Column(data=2 * np.sqrt(np.log(2) * (objects['a']**2 + objects['b']**2)),
                              name='fwhm_custom'))

    # Measure R30, R50, R80
    temp = sep.flux_radius(
        input_data, objects['x'], objects['y'], 6. * objects['a'], [0.3, 0.5, 0.8])[0]
    objects.add_column(Column(data=temp[:, 0], name='R30'))
    objects.add_column(Column(data=temp[:, 1], name='R50'))
    objects.add_column(Column(data=temp[:, 2], name='R80'))

    # Use Kron radius to calculate FLUX_AUTO in SourceExtractor.
    # Here PHOT_PARAMETER = 2.5, 3.5
    if flux_auto:
        kronrad, krflag = sep.kron_radius(input_data, objects['x'], objects['y'],
                                          objects['a'], objects['b'],
                                          objects['theta'], 6.0)
        flux, fluxerr, flag = sep.sum_circle(input_data, objects['x'], objects['y'],
                                             2.5 * (kronrad), subpix=1)
        flag |= krflag  # combine flags into 'flag'

        r_min = 1.75  # minimum diameter = 3.5
        use_circle = kronrad * np.sqrt(objects['a'] * objects['b']) < r_min
        cflux, cfluxerr, cflag = sep.sum_circle(input_data, objects['x'][use_circle], objects['y'][use_circle],
                                                r_min, subpix=1)
        flux[use_circle] = cflux
        fluxerr[use_circle] = cfluxerr
        flag[use_circle] = cflag
        objects.add_column(Column(data=flux, name='flux_auto'))
        objects.add_column(Column(data=kronrad, name='kron_rad'))

    if flux_aper is not None:
        if len(flux_aper) != 2:
            raise ValueError('"flux_aper" must be a list with length = 2.')
        objects.add_column(Column(data=sep.sum_circle(input_data, objects['x'], objects['y'], flux_aper[0])[0],
                                  name='flux_aper_1'))
        objects.add_column(Column(data=sep.sum_circle(input_data, objects['x'], objects['y'], flux_aper[1])[0],
                                  name='flux_aper_2'))
        objects.add_column(Column(data=sep.sum_circann(input_data, objects['x'], objects['y'],
                                                       flux_aper[0], flux_aper[1])[0], name='flux_ann'))

    # plot background-subtracted image
    if show_fig:
        fig, ax = plt.subplots(1, 2, figsize=(12, 6))
        if min(input_data.shape) * pixel_scale < 30:
            scale_bar_length = 5
        elif min(input_data.shape) * pixel_scale > 100:
            scale_bar_length = 61
        else:
            scale_bar_length = 10
        ax[0] = display_single(
            input_data, ax=ax[0], scale_bar_length=scale_bar_length, pixel_scale=pixel_scale)
        if mask is not None:
            ax[0].imshow(mask.astype(float), origin='lower',
                         alpha=0.1, cmap='Greys_r')
        from matplotlib.patches import Ellipse
        # plot an ellipse for each object
        for obj in objects:
            e = Ellipse(xy=(obj['x'], obj['y']),
                        width=5 * obj['a'],
                        height=5 * obj['b'],
                        angle=obj['theta'] * 180. / np.pi)
            e.set_facecolor('none')
            e.set_edgecolor('red')
            ax[0].add_artist(e)
        ax[1] = display_single(segmap, scale='linear', cmap=SEG_CMAP,
                               ax=ax[1], scale_bar_length=scale_bar_length)
        # plt.savefig('./extract_obj.png', bbox_inches='tight')
        return objects, segmap, fig
    return objects, segmap


def _image_gaia_stars_tigress(image, wcs, pixel_scale=0.168, mask_a=694.7, mask_b=3.5,
                              verbose=True, visual=False, size_buffer=1.4, logger=None):
    """
    Search for bright stars using GAIA catalogs on Tigress (`/tigress/HSC/refcats/htm/gaia_dr2_20200414`).
    For more information, see https://community.lsst.org/t/gaia-dr2-reference-catalog-in-lsst-format/3901.
    This function requires `lsstpipe`.

    Parameters:
        image (numpy 2-D array): input image.
        wcs (`astropy.wcs` object): WCS of the input image.
        pixel_scale (float): default is 0.168 (HSC pixel size). This only affect the figure scale bar.
        mask_a (float): a scaling factor for the size of the plotted star, larger value means larger circle will be plotted.
        mask_b (float): a scale size for the plotted star, larger value gives larger circle. 
        visual (bool): whether display the matched Gaia stars.

    Return: 
        gaia_results (`astropy.table.Table` object): a catalog of matched stars.
    """
    # Central coordinate
    ra_cen, dec_cen = wcs.all_pix2world(image.shape[1] / 2,
                                        image.shape[0] / 2,
                                        0)
    img_cen_ra_dec = SkyCoord(
        ra_cen, dec_cen, unit=('deg', 'deg'), frame='icrs')

    # Width and height of the search box
    img_ra_size = Quantity(pixel_scale * (image.shape)
                           [1] * size_buffer, u.arcsec).to(u.degree)
    img_dec_size = Quantity(pixel_scale * (image.shape)
                            [0] * size_buffer, u.arcsec).to(u.degree)

    # Search for stars in Gaia catatlogs, which are stored in
    # `/tigress/HSC/refcats/htm/gaia_dr2_20200414`.
    try:
        from lsst.meas.algorithms.htmIndexer import HtmIndexer
        import lsst.geom as geom

        def getShards(ra, dec, radius):
            htm = HtmIndexer(depth=7)

            afw_coords = geom.SpherePoint(
                geom.Angle(ra, geom.degrees),
                geom.Angle(dec, geom.degrees))

            shards, onBoundary = htm.getShardIds(
                afw_coords, radius * geom.degrees)
            return shards

    except ImportError as e:
        # Output expected ImportErrors.
        if logger is not None:
            logger.error(e)
            logger.error(
                'LSST Pipe must be installed to query Gaia stars on Tigress.')
        print(e)
        print('LSST Pipe must be installed to query Gaia stars on Tigress.')

    # find out the Shard ID of target area in the HTM (Hierarchical triangular mesh) system
    if logger is not None:
        logger.info('    Taking Gaia catalogs stored in `Tigress`')
    print('    Taking Gaia catalogs stored in `Tigress`')

    shards = getShards(ra_cen, dec_cen, max(
        img_ra_size, img_dec_size).to(u.degree).value)
    cat = vstack([Table.read(
        f'/tigress/HSC/refcats/htm/gaia_dr2_20200414/{index}.fits') for index in shards])
    cat['coord_ra'] = cat['coord_ra'].to(u.degree)
    # why GAIA coordinates are in RADIAN???
    cat['coord_dec'] = cat['coord_dec'].to(u.degree)

    # Trim this catalog a little bit
    # Ref: https://github.com/MerianSurvey/caterpillar/blob/main/caterpillar/catalog.py
    if cat:  # if not empty
        gaia_results = cat[
            (cat['coord_ra'] > img_cen_ra_dec.ra - img_ra_size / 2) &
            (cat['coord_ra'] < img_cen_ra_dec.ra + img_ra_size / 2) &
            (cat['coord_dec'] > img_cen_ra_dec.dec - img_dec_size / 2) &
            (cat['coord_dec'] < img_cen_ra_dec.dec + img_dec_size / 2)
        ]
        gaia_results.rename_columns(['coord_ra', 'coord_dec'], ['ra', 'dec'])

        gaia_results['phot_g_mean_mag'] = -2.5 * \
            np.log10(
                (gaia_results['phot_g_mean_flux'] / (3631 * u.Jy)))  # AB magnitude

        # Convert the (RA, Dec) of stars into pixel coordinate using WCS
        x_gaia, y_gaia = wcs.all_world2pix(gaia_results['ra'],
                                           gaia_results['dec'], 0)

        # Generate mask for each star
        rmask_gaia_arcsec = mask_a * np.exp(
            -gaia_results['phot_g_mean_mag'] / mask_b)

        # Update the catalog
        gaia_results.add_column(Column(data=x_gaia, name='x_pix'))
        gaia_results.add_column(Column(data=y_gaia, name='y_pix'))
        gaia_results.add_column(
            Column(data=rmask_gaia_arcsec, name='rmask_arcsec'))

        if visual:
            fig = plt.figure(figsize=(8, 8))
            ax1 = fig.add_subplot(111)

            ax1 = display_single(image, ax=ax1)
            # Plot an ellipse for each object
            for star in gaia_results:
                smask = mpl_ellip(
                    xy=(star['x_pix'], star['y_pix']),
                    width=(2.0 * star['rmask_arcsec'] / pixel_scale),
                    height=(2.0 * star['rmask_arcsec'] / pixel_scale),
                    angle=0.0)
                smask.set_facecolor(ORG(0.2))
                smask.set_edgecolor(ORG(1.0))
                smask.set_alpha(0.3)
                ax1.add_artist(smask)

            # Show stars
            ax1.scatter(
                gaia_results['x_pix'],
                gaia_results['y_pix'],
                color=ORG(1.0),
                s=100,
                alpha=0.9,
                marker='+')

            ax1.set_xlim(0, image.shape[1])
            ax1.set_ylim(0, image.shape[0])

        return gaia_results

    return None


def image_gaia_stars(image, wcs, pixel_scale=0.168, mask_a=694.7, mask_b=3.5,
                     verbose=False, visual=False, size_buffer=1.4, tap_url=None):
    """
    Search for bright stars using GAIA catalog. From https://github.com/dr-guangtou/kungpao.

    Parameters:
        image (numpy 2-D array): input image.
        wcs (`astropy.wcs` object): WCS of the input image.
        pixel_scale (float): default is 0.168 (HSC pixel size). This only affect the figure scale bar.
        mask_a (float): a scaling factor for the size of the plotted star, larger value means larger circle will be plotted.
        mask_b (float): a scale size for the plotted star, larger value gives larger circle. 
        visual (bool): whether display the matched Gaia stars.

    Return: 
        gaia_results (`astropy.table.Table` object): a catalog of matched stars.
    """
    # Central coordinate
    ra_cen, dec_cen = wcs.all_pix2world(image.shape[0] / 2,
                                        image.shape[1] / 2,
                                        0)
    img_cen_ra_dec = SkyCoord(
        ra_cen, dec_cen, unit=('deg', 'deg'), frame='icrs')

    # Width and height of the search box
    img_search_x = Quantity(pixel_scale * (image.shape)
                            [0] * size_buffer, u.arcsec)
    img_search_y = Quantity(pixel_scale * (image.shape)
                            [1] * size_buffer, u.arcsec)

    # Search for stars
    if tap_url is not None:
        with suppress_stdout():
            from astroquery.gaia import TapPlus, GaiaClass
            Gaia = GaiaClass(TapPlus(url=tap_url))

            gaia_results = Gaia.query_object_async(
                coordinate=img_cen_ra_dec,
                width=img_search_x,
                height=img_search_y,
                verbose=verbose)
    else:
        with suppress_stdout():
            from astroquery.gaia import Gaia

            gaia_results = Gaia.query_object_async(
                coordinate=img_cen_ra_dec,
                width=img_search_x,
                height=img_search_y,
                verbose=verbose)

    if gaia_results:
        # Convert the (RA, Dec) of stars into pixel coordinate
        ra_gaia = np.asarray(gaia_results['ra'])
        dec_gaia = np.asarray(gaia_results['dec'])
        x_gaia, y_gaia = wcs.all_world2pix(ra_gaia, dec_gaia, 0)

        # Generate mask for each star
        rmask_gaia_arcsec = mask_a * np.exp(
            -gaia_results['phot_g_mean_mag'] / mask_b)

        # Update the catalog
        gaia_results.add_column(Column(data=x_gaia, name='x_pix'))
        gaia_results.add_column(Column(data=y_gaia, name='y_pix'))
        gaia_results.add_column(
            Column(data=rmask_gaia_arcsec, name='rmask_arcsec'))

        if visual:
            fig = plt.figure(figsize=(8, 8))
            ax1 = fig.add_subplot(111)

            ax1 = display_single(image, ax=ax1)
            # Plot an ellipse for each object
            for star in gaia_results:
                smask = mpl_ellip(
                    xy=(star['x_pix'], star['y_pix']),
                    width=(2.0 * star['rmask_arcsec'] / pixel_scale),
                    height=(2.0 * star['rmask_arcsec'] / pixel_scale),
                    angle=0.0)
                smask.set_facecolor(ORG(0.2))
                smask.set_edgecolor(ORG(1.0))
                smask.set_alpha(0.3)
                ax1.add_artist(smask)

            # Show stars
            ax1.scatter(
                gaia_results['x_pix'],
                gaia_results['y_pix'],
                color=ORG(1.0),
                s=100,
                alpha=0.9,
                marker='+')

            ax1.set_xlim(0, image.shape[0])
            ax1.set_ylim(0, image.shape[1])

        return gaia_results

    return None


def gaia_star_mask(img, wcs, gaia_stars=None, pixel_scale=0.168, mask_a=694.7, mask_b=3.5,
                   size_buffer=1.4, gaia_bright=18.0,
                   factor_b=1.3, factor_f=1.9, tigress=False, logger=None):
    """Find stars using Gaia and mask them out if necessary. From https://github.com/dr-guangtou/kungpao.

    Using the stars found in the GAIA TAP catalog, we build a bright star mask following
    similar procedure in Coupon et al. (2017).

    We separate the GAIA stars into bright (G <= 18.0) and faint (G > 18.0) groups, and
    apply different parameters to build the mask.

    Parameters:
        img (numpy 2-D array): input image.
        wcs (`astropy.wcs` object): WCS of the input image.
        pixel_scale (float): default is 0.168 (HSC pixel size). This only affect the figure scale bar.
        mask_a (float): a scale factor for the size of the plotted star, larger value means larger circle will be plotted.
        mask_b (float): a scale size for the plotted star, larger value gives larger circle. 
        gaia_bright (float): a threshold above which are classified as bright stars.
        factor_b (float): a scale size of mask for bright stars. Larger value gives smaller mask size.
        factor_f (float): a scale size of mask for faint stars. Larger value gives smaller mask size.
        tigress (bool): whether take Gaia catalogs on Tigress

    Return: 
        msk_star (numpy 2-D array): the masked pixels are marked by one.

    """
    if gaia_stars is None:
        if tigress:
            gaia_stars = _image_gaia_stars_tigress(img, wcs, pixel_scale=pixel_scale,
                                                   mask_a=mask_a, mask_b=mask_b,
                                                   verbose=False, visual=False,
                                                   size_buffer=size_buffer, logger=logger)
        else:
            gaia_stars = image_gaia_stars(img, wcs, pixel_scale=pixel_scale,
                                          mask_a=mask_a, mask_b=mask_b,
                                          verbose=False, visual=False,
                                          size_buffer=size_buffer)
        gaia_stars = gaia_stars[(abs(gaia_stars['pm_ra']) +
                                 abs(gaia_stars['pm_dec']) + gaia_stars['parallax'] != 0)]
        if len(gaia_stars) == 0:
            gaia_stars = None

        if gaia_stars is not None:
            if logger is not None:
                logger.info(
                    f'    {len(gaia_stars)} stars from Gaia are masked!')
            print(f'    {len(gaia_stars)} stars from Gaia are masked!')
        else:  # does not find Gaia stars
            if logger is not None:
                logger.info('    No Gaia stars are masked.')
            print('    No Gaia stars are masked.')
    else:
        if logger is not None:
            logger.info(f'    {len(gaia_stars)} stars from Gaia are masked!')
        print(f'    {len(gaia_stars)} stars from Gaia are masked!')

    # Make a mask image
    msk_star = np.zeros(img.shape).astype('uint8')

    # Remove sources with no parallax and proper motion
    if gaia_stars is not None:
        gaia_b = gaia_stars[gaia_stars['phot_g_mean_mag'] <= gaia_bright]
        sep.mask_ellipse(msk_star, gaia_b['x_pix'], gaia_b['y_pix'],
                         gaia_b['rmask_arcsec'] / factor_b / pixel_scale,
                         gaia_b['rmask_arcsec'] / factor_b / pixel_scale, 0.0, r=1.0)

        gaia_f = gaia_stars[gaia_stars['phot_g_mean_mag'] > gaia_bright]
        sep.mask_ellipse(msk_star, gaia_f['x_pix'], gaia_f['y_pix'],
                         gaia_f['rmask_arcsec'] / factor_f / pixel_scale,
                         gaia_f['rmask_arcsec'] / factor_f / pixel_scale, 0.0, r=1.0)

        return gaia_stars, msk_star

    return None, msk_star


def padding_PSF(psf_list):
    '''
    If the sizes of HSC PSF in all bands are not the same, this function pads the smaller PSFs.

    Parameters:
        psf_list: a list returned by `unagi.task.hsc_psf` function

    Returns:
        psf_pad: a list including padded PSFs. They now share the same size.
    '''
    # Padding PSF cutouts from HSC
    max_len = max([max(psf[0].data.shape) for psf in psf_list])
    psf_pad = []
    for psf in psf_list:
        y_len, x_len = psf[0].data.shape
        dy = (max_len - y_len) // 2
        dx = (max_len - x_len) // 2
        temp = np.pad(psf[0].data.astype('float'), ((dy, dy),
                                                    (dx, dx)), 'constant', constant_values=0)
        if temp.shape == (max_len, max_len):
            psf_pad.append(temp)
        else:
            raise ValueError('Wrong size!')

    return psf_pad
# usage:
# python imageStitcher2.py \
#   -o outDir \
#   calexp-HSC-I-937[0-3]-*,*.fits
import lsst.afw.image as afwImage
import lsst.afw.math as afwMath
import lsst.afw.geom as afwGeom
import lsst.geom as lsstGeom
import argparse
import logging
import multiprocessing
import numpy
import os
import astropy.io.fits as afits


def main():
    parser = argparse.ArgumentParser(description='This tool stitches fits images with non-unique projection into a large image')
    parser.add_argument('--out', '-o', required=True, help='output directory')
    parser.add_argument('--crval', type=float, nargs=2, metavar=('CRVAL1', 'CRVAL2'), help='CRVAL of output image')
    parser.add_argument('--pixel-scale', type=lambda a: float(a) / 3600., default=0.168 / 3600., help='pixel scale in arcsec')
    parser.add_argument('--scale', type=float, default=1, help='scale for the output image')
    parser.add_argument('--patch-size', type=int, default=4000, help='patch size for intermediate processing')
    parser.add_argument('--parallel', '-j', type=int, metavar='NUMBER_OR_PROCESSES', help='the number of jobs to run simltaneously')
    parser.add_argument('exposure', nargs='+', metavar='FILE', help='src files (must have WCS)')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

    os.makedirs(args.out, exist_ok=True)

    warperConfig = afwMath.WarperConfig()

    projection = Projection(args.exposure, crval=args.crval, pixelScale=args.pixel_scale / args.scale)
    wcs = projection.targetWcs()

    mapArgs = []
    warpFiles = []
    for xi, yi, bbox, files in projection.patches(int(args.patch_size * args.scale)):
        outFile = f'{args.out}/{xi}-{yi}.fits'
        warpFiles.append(outFile)
        mapArgs.append((warperConfig, wcs, bbox, outFile, files))

    multiprocessing.Pool(args.parallel).map(argsPackedWarp, mapArgs)

    bbox = boundary(warpFiles)

    afits.HDUList([
        stitchedHdu(warpFiles, bbox, image_index=1),
        # stitchedHdu(warpFiles, bbox, image_index=2, dtype=numpy.uint16),
        # stitchedHdu(warpFiles, bbox, image_index=3),
    ]).writeto(args.out + '/stitched.fits', output_verify='silentfix', overwrite=True)

    logging.warning('This program is not fully tested. Use resultant data carefully')


def warp(config, targetWcs, bbox, outFile, files):
    if len(files) == 0:
        return
    warper = afwMath.Warper.fromConfig(config)
    for i, f in enumerate(files):
        logging.info('warping {} -> {}...'.format(f, outFile))
        e = afwImage.ExposureF(f)
        w1 = warper.warpExposure(targetWcs, e, destBBox=bbox)
        if i == 0:
            w = eraseNoDataArea(w1)
        else:
            mergeExposure(w, w1)
    w.writeFits(outFile)


def argsPackedWarp(args):
    return warp(*args)


def eraseNoDataArea(exp):
    mi = exp.getMaskedImage()
    nodataValue = 1 << mi.getMask().getMaskPlane('NO_DATA')
    ng = mi.getMask().getArray() & nodataValue > 0
    mi.getImage().getArray()[ng] = numpy.nan
    return exp


def mergeExposure(a, b):
    a_mi = a.getMaskedImage()
    b_mi = b.getMaskedImage()
    nodataValue = 1 << b_mi.getMask().getMaskPlane('NO_DATA')
    ok = b_mi.getMask().getArray() & nodataValue == 0
    a_mi.getImage().getArray()[ok] = b_mi.getImage().getArray()[ok]
    a_mi.getMask().getArray()[ok] = b_mi.getMask().getArray()[ok]
    a_mi.getVariance().getArray()[ok] = b_mi.getVariance().getArray()[ok]


class Projection(object):
    def __init__(self, fnames, pixelScale, crval=None):
        self.fnames = fnames
        self.crval = crval or self._calcCrval()
        self.pixelScale = pixelScale
        self._polygons = self._buildPolygons()


    def targetWcs(self):
        crpix = lsstGeom.Point2D(0., 0.)
        crval = lsstGeom.SpherePoint(*self.crval, afwGeom.degrees)
        cdMatrix = numpy.array([[-self.pixelScale, 0.], [0., self.pixelScale]])
        return afwGeom.makeSkyWcs(crpix, crval, cdMatrix)


    def patches(self, patchSize):
        w = patchSize
        ((minX, minY), (maxX, maxY)) = self._getBBox()
        for xi, x in enumerate(range(int(minX), int(maxX + 1), patchSize)):
            for yi, y in enumerate(range(int(minY), int(maxY + 1), patchSize)):
                overlapFnames = []
                patchPolygon = afwGeom.polygon.Polygon([
                    lsstGeom.Point2D(x + w * xx, y + w * yy)
                    for xx, yy in [(0., 0.), (1., 0.), (1., 1.), (0., 1.)]
                ])
                for i, p in enumerate(self._polygons):
                    if p.overlaps(patchPolygon):
                        overlapFnames.append(self.fnames[i])
                bbox = lsstGeom.Box2I(lsstGeom.Point2I(x, y), lsstGeom.Extent2I(w, w))
                yield xi, yi, bbox, overlapFnames


    def _getBBox(self):
        b = lsstGeom.Box2D()
        for p in self._polygons:
            b.include(p.getBBox())
        return b.getMin(), b.getMax()


    def _calcCrval(self):
        centerCoords = []
        for fname in self.fnames:
            md = afwImage.readMetadata(fname)
            wcs = afwGeom.makeSkyWcs(md)
            x = md.get('ZNAXIS1') / 2. - md.get('LTV1')
            y = md.get('ZNAXIS2') / 2. - md.get('LTV2')
            centerCoords.append([c.asDegrees() for c in wcs.pixelToSky(x, y)])
        centerCoords = numpy.deg2rad(centerCoords)
        A, D = numpy.vstack(centerCoords).T
        X, Y, Z = numpy.array(ad2xyz(A, D)).mean(axis=1)
        return numpy.rad2deg(xyz2ad(X, Y, Z))


    def _buildPolygons(self):
        cornerCoords = []
        for fname in self.fnames:
            md = afwImage.readMetadata(fname)
            wcs = afwGeom.makeSkyWcs(md)
            w = md.get('ZNAXIS1')
            h = md.get('ZNAXIS2')
            cornerPixels = [(x - md.get('LTV1'), y - md.get('LTV2')) for x, y in [(0, 0), (w, 0), (w, h), (0, h)]]
            cornerCoords += [[c.asRadians() for c in wcs.pixelToSky(x, y)] for x, y in cornerPixels]
        cornerXYZ = ad2xyz(*numpy.vstack(cornerCoords).T)

        refPoint = ad2xyz(*numpy.deg2rad(self.crval))
        w = numpy.cross(refPoint, [0, 0, 1]) # west axis
        w /= numpy.linalg.norm(w)
        n = numpy.cross(w, refPoint)         # north axis
        n /= numpy.linalg.norm(n)

        T = numpy.inner(w, numpy.array(cornerXYZ).T) / numpy.deg2rad(self.pixelScale)
        U = numpy.inner(n, numpy.array(cornerXYZ).T) / numpy.deg2rad(self.pixelScale)

        polygons = []
        for i in range(len(self.fnames)):
            p = afwGeom.polygon.Polygon([lsstGeom.Point2D(T[4 * i + j], U[4 * i + j]) for j in range(4)])
            polygons.append(p)

        return polygons


def ad2xyz(a, d):
    cosd = numpy.cos(d)
    return (
        cosd * numpy.cos(a),
        cosd * numpy.sin(a),
        numpy.sin(d),
    )


def xyz2ad(x, y, z):
    r = numpy.sqrt(x*x + y*y)
    a = numpy.arctan2(y, x)
    d = numpy.arctan2(z, r)
    return a % (2 * numpy.pi), d



def stitchedHdu(files, boundary, nodata=float('nan'), meta_index=0, image_index=1, dtype='float32', hduFactory=afits.ImageHDU):
    #        ^
    #        |
    #        |
    #   +----+----------------+
    #   |    |             (maxx, maxy)
    #   | +--+-------+        |
    #   | |  |    (naxis1-crpix1, naxis2-crpix2)
    #   | |  |       |        |
    #---|-+--O-------+--------+--->
    #   | |  |       |        |
    #   | +--+-------+        |
    #   |(-crpix1, -crpix2)   |
    #   +----+----------------+
    # (minx, miny)
    #

    ((minx, miny), (maxx, maxy)) = boundary

    width = int(maxx - minx)
    height = int(maxy - miny)

    logging.info('allocating image buffer %(width)d x %(height)d' % locals())
    pool = numpy.empty((height, width), dtype=dtype)
    pool.fill(nodata)

    fluxMag0 = None
    for fname in files:
        logging.info('pasting %(fname)s...' % locals())
        try:
            with afits.open(fname) as hdul:
                if fluxMag0 is None and 'FLUXMAG0' in hdul[meta_index].header:
                    fluxMag0 = hdul[0].header['FLUXMAG0']
                header = hdul[image_index].header
                data = hdul[image_index].data
                crpix1 = int(header['CRPIX1'])
                crpix2 = int(header['CRPIX2'])
                naxis1 = int(header['NAXIS1'])
                naxis2 = int(header['NAXIS2'])
                pool[-crpix2 - miny : naxis2 - crpix2 - miny,
                     -crpix1 - minx : naxis1 - crpix1 - minx] = data
        except:
            import traceback
            traceback.print_exc()
            # logging.info('failed to read %s' % fname)

    hdu = hduFactory(pool)
    header['CRPIX1'] = -minx
    header['CRPIX2'] = -miny
    # header['LTV1'] = header['CRPIX1'] - ref_point_phys1
    # header['LTV2'] = header['CRPIX2'] - ref_point_phys2
    # if fluxMag0 is not None:
    #     header['FLUXMAG0'] = fluxMag0
    hdu.header = header

    return hdu


def boundary(files, image_index=1):
    #    ^
    #    |    +---------+
    #    |    |        (X,Y)
    #    |    |         |
    #    |    +---------+
    #    |   (x,y)
    #----O------------------->
    #    |

    logging.info('setting stitched image boundary.')
    minx = []
    miny = []
    maxx = []
    maxy = []
    for fname in files:
        logging.info('reading header of %(fname)s...' % locals())
        try:
            with afits.open(fname) as hdul:
                header = hdul[image_index].header
                minx.append(-header['CRPIX1'])
                miny.append(-header['CRPIX2'])
                maxx.append(-header['CRPIX1'] + header['NAXIS1'])
                maxy.append(-header['CRPIX2'] + header['NAXIS2'])
        except:
            import traceback
            traceback.print_exc()
    return (int(min(minx)), int(min(miny))), (int(max(maxx)), int(max(maxy)))


if __name__ == '__main__':
    main()

# usage:
#   python imageStitcher1.py -o ngc4030-stitch.fits ngc4030-out/deepCoadd/HSC-G/0/*/calexp-HSC-G-0-*

import numpy
from astropy.io import fits as afits
import logging
import logging
import traceback
import itertools


def stitchedHdu(files, boundary, *, nodata=float('nan'), meta_index=0, image_index=1, dtype='float32', binsize, hduFactory=afits.ImageHDU):
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

    bin = Bin(binsize)

    ((minx, miny), (maxx, maxy)) = boundary
    minx = bin(minx)
    miny = bin(miny)
    maxx = bin(maxx)
    maxy = bin(maxy)

    width = maxx - minx
    height = maxy - miny

    logging.info(f'allocating image buffer {width} x {height}')
    pool = numpy.empty((height, width), dtype=dtype)
    pool.fill(nodata)

    baseFluxMag0 = 10 ** (27 / 2.5)

    for fname in files:
        logging.info('pasting %(fname)s...' % locals())
        try:
            with afits.open(fname) as hdul:
                fluxMag0, fluxMag0Err = getFluxMag0(hdul)
                header = hdul[image_index].header
                data = binimage(hdul[image_index].data, binsize)
                for i, j in itertools.product([1, 2], [1, 2]):
                    key = f'CD{i}_{j}'
                    if key in header:
                        header[key] *= binsize
            assert float.is_integer(header['CRPIX1'])
            assert float.is_integer(header['CRPIX2'])
            crpix1 = bin(int(header['CRPIX1']))
            crpix2 = bin(int(header['CRPIX2']))
            naxis1 = bin(header['NAXIS1'])
            naxis2 = bin(header['NAXIS2'])
            pool[-crpix2 - miny : naxis2 - crpix2 - miny,
                 -crpix1 - minx : naxis1 - crpix1 - minx] = (baseFluxMag0 / fluxMag0) * data
        except:
            traceback.print_exc()
    
    header['FLUXMAG0'] = baseFluxMag0

    hdu = hduFactory(pool)
    header['CRPIX1'] = (header['CRPIX1'] - 0.5) / binsize + 0.5
    header['CRPIX2'] = (header['CRPIX2'] - 0.5) / binsize + 0.5
    header['LTV1'] += -header['CRPIX1'] - minx
    header['LTV2'] += -header['CRPIX2'] - miny
    header['CRPIX1'] = -minx
    header['CRPIX2'] = -miny
    hdu.header = header

    return hdu


class Bin:
    def __init__(self, binsize):
        self._binsize = binsize

    def __call__(self, n):
        assert n % self._binsize == 0, f'binsize should be one of divisor of {n}'
        return n // self._binsize


def binimage(array, binsize):
    if binsize == 1:
        return array
    else:
        s = array.shape
        b = binsize
        assert s[0] % b == 0, f'binsize should be one of divisor of image height ({s[0]})'
        assert s[1] % b == 0, f'binsize should be one of divisor of image width ({s[1]})'
        array2 = numpy.average(array.reshape((s[0] // b, b, s[1] // b, b)), axis=(1, 3))
        return array2


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
        with afits.open(fname) as hdul:
            header = hdul[image_index].header
            minx.append(int(-header['CRPIX1']))
            miny.append(int(-header['CRPIX2']))
            maxx.append(int(-header['CRPIX1'] + header['NAXIS1']))
            maxy.append(int(-header['CRPIX2'] + header['NAXIS2']))
    return (min(minx), min(miny)), (max(maxx), max(maxy))


def getFluxMag0(hdus):
    if 'FLUXMAG0' in hdus[0].header:
        return hdus[0].header['FLUXMAG0'], float('nan')
    else:
        entryHduIndex = hdus[0].header["AR_HDU"] - 1
        entryHdu = hdus[entryHduIndex]
        photoCalibId = hdus[0].header["PHOTOCALIB_ID"]
        photoCalibEntry, = entryHdu.data[entryHdu.data["id"] == photoCalibId]
        photoCalibHdu = hdus[entryHduIndex + photoCalibEntry["cat.archive"]]
        start = photoCalibEntry["row0"]
        end = start + photoCalibEntry["nrows"]
        photoCalib, = photoCalibHdu.data[start:end]
        calibrationMean = photoCalib["calibrationMean"]
        calibrationErr  = photoCalib["calibrationErr"]
        if calibrationMean != 0.0:
            fluxMag0 = (1.0e+23 * 10**(48.6 / (-2.5)) * 1.0e+9) / calibrationMean
            fluxMag0Err = (1.0e+23 * 10**(48.6 / (-2.5)) * 1.0e+9) / calibrationMean**2 * calibrationErr
        else:
            fluxMag0 = float('nan')
            fluxMag0Err = float('nan')
        return fluxMag0, fluxMag0Err


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='This tool stitches adjacent patches in the same tract together.')
    parser.add_argument('--out', '-o', required=True, help='output file')
    parser.add_argument('--binsize', '-b', type=int, default=1, help='bin size')
    parser.add_argument('files', nargs='+', metavar='FILE', help='patch files to be stitched')
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

    boundary = boundary(args.files)
    imageHdu = stitchedHdu(args.files, boundary, binsize=args.binsize, hduFactory=afits.PrimaryHDU)
    # maskHdu  = stitchedHdu(args.files, boundary, image_index=2, dtype='uint16')
    # afits.HDUList([imageHdu, maskHdu]).writeto(args.out, output_verify='fix', overwrite=True)
    afits.HDUList([imageHdu]).writeto(args.out, output_verify='fix', overwrite=True)

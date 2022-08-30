# Image Stitcher #2
Similar to [Image Stitcher #1](../imageStitcher1), but this one allows you to stitch patches in different tracts together.  The HSC (or LSST) pipeline has to be installed to use this tool.  The overlapping regions between the adjacent tracts are not exactly the same; tract A may have slightly different DNs and astrometry from tract B in the overlapping region.  This tool simply adopts the pixels from tract B in the resultant image.  This tool can also be used for CCD images to visualize a full focal plane image from a visit.

## Prerequisites
* hscPipe 7.9.1

## Usage
* For Patch Files
```sh
python imageStitcher2.py -o product ./pdr1_wide/deepCoadd-results/HSC-I/852[45]/*,*/calexp-*.fits

ds9 product/stitched.fits
```
* For CORR Files
```sh
python imageStitcher2.py -o product outputs/single-frame-driver/01232/HSC-G/corr/CORR-0029414-*.fits

ds9 product/stitched.fits
```

## ```--help```
```
usage: imageStitcher2.py [-h] --out OUT [--crval CRVAL1 CRVAL2]
                         [--pixel-scale PIXEL_SCALE] [--scale SCALE]
                         [--patch-size PATCH_SIZE]
                         [--parallel NUMBER_OR_PROCESSES]
                         FILE [FILE ...]

This tool stitches fits images with non-unique projection into a large image

positional arguments:
  FILE                  src files (must have WCS)

optional arguments:
  -h, --help            show this help message and exit
  --out OUT, -o OUT     output directory
  --crval CRVAL1 CRVAL2
                        CRVAL of output image
  --pixel-scale PIXEL_SCALE
                        pixel scale in arcsec
  --scale SCALE         scale for the output image
  --patch-size PATCH_SIZE
                        patch size for intermediate processing
  --parallel NUMBER_OR_PROCESSES, -j NUMBER_OR_PROCESSES
                        the number of jobs to run simltaneously
```